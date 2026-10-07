"""Independent garment-classifier evaluation on labelled photos or Fashionpedia crops.

The personal-photo manifest is never used for fitting or early stopping. The
Fashionpedia mode is a separate, natural-photo check that can run immediately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from fashion_ai.classifier import LABELS_PATH, MODEL_PATH, _load_model, image_array


LABELS = ("top", "bottom", "shoes", "dress")


def manifest_examples(path: Path) -> tuple[list[tuple[Image.Image, str]], dict]:
    examples = []
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("The manifest needs image,label columns.")
        for number, row in enumerate(reader, 2):
            label = (row["label"] or "").strip().lower()
            if label not in LABELS:
                raise ValueError(f"Line {number}: label must be one of {', '.join(LABELS)}.")
            image_path = (path.parent / (row["image"] or "").strip()).resolve()
            if image_path in seen or not image_path.is_file():
                raise ValueError(f"Line {number}: image is missing or listed twice: {image_path}")
            seen.add(image_path)
            with Image.open(image_path) as source:
                examples.append((source.convert("RGB").copy(), label))
    if not examples:
        raise ValueError("Add at least one labelled image to the manifest.")
    return examples, {"source": "User-labelled personal clothing photos", "manifest": str(path), "image_count": len(examples)}


def fashionpedia_examples(images_path: Path, annotations_path: Path, per_class: int, seed: int):
    import pyarrow.parquet as parquet

    from evaluate_fashionpedia import FASHIONPEDIA_TO_APP, instance_mask
    from fashion_ai.segmentation import LABELS as APP_LABELS

    source = json.loads(annotations_path.read_text(encoding="utf-8"))
    candidates = defaultdict(list)
    for annotation in source["annotations"]:
        label_id = FASHIONPEDIA_TO_APP.get(annotation["category_id"])
        if label_id is None or annotation.get("iscrowd"):
            continue
        x, y, width, height = annotation["bbox"]
        if width >= 45 and height >= 45:
            label = APP_LABELS[label_id]
            candidates["top" if label == "outerwear" else label].append(annotation)
    rng = random.Random(seed)
    selected = []
    used_images = set()
    for label in LABELS:
        choices = candidates[label][:]
        rng.shuffle(choices)
        for annotation in choices:
            if annotation["image_id"] in used_images:
                continue
            selected.append((annotation, label))
            used_images.add(annotation["image_id"])
            if sum(chosen_label == label for _, chosen_label in selected) == per_class:
                break
        else:
            raise ValueError(f"Only found {sum(chosen_label == label for _, chosen_label in selected)} usable {label} photos.")
    needed = {annotation["image_id"] for annotation, _ in selected}
    image_bytes = {}
    for batch in parquet.ParquetFile(images_path).iter_batches(batch_size=32, columns=["image_id", "image"]):
        for row in batch.to_pylist():
            if row["image_id"] in needed:
                image_bytes[row["image_id"]] = row["image"]["bytes"]
        if len(image_bytes) == len(needed):
            break
    if len(image_bytes) != len(needed):
        raise ValueError("Some selected Fashionpedia images are missing from the parquet.")
    context, isolated = [], []
    for annotation, label in selected:
        with Image.open(io.BytesIO(image_bytes[annotation["image_id"]])) as source_image:
            image = source_image.convert("RGB")
        x, y, width, height = annotation["bbox"]
        pad = max(width, height) * 0.08
        box = (max(0, math.floor(x - pad)), max(0, math.floor(y - pad)),
               min(image.width, math.ceil(x + width + pad)), min(image.height, math.ceil(y + height + pad)))
        crop = image.crop(box)
        mask = instance_mask(annotation, image.height, image.width)[box[1]:box[3], box[0]:box[2]]
        pixels = np.asarray(crop).copy()
        pixels[~mask] = 255
        context.append((crop, label))
        isolated.append((Image.fromarray(pixels), label))
    metadata = {
        "source": "Fashionpedia official validation split; single-garment instance crops from natural photos",
        "seed": seed,
        "per_class": per_class,
        "image_count": len(selected),
        "image_ids": [annotation["image_id"] for annotation, _ in selected],
        "annotation_ids": [annotation["id"] for annotation, _ in selected],
        "notes": ["These are annotated fashion photos, not the user's own clothing photos.",
                  "Context crops can show other garments; isolated crops use the Fashionpedia instance mask and a white background."],
    }
    return context, isolated, metadata


def metrics(model, labels: list[str], examples: list[tuple[Image.Image, str]]) -> dict:
    probability_batches = []
    inference_seconds = 0.0
    for start in range(0, len(examples), 32):
        batch = np.stack([image_array(image) for image, _ in examples[start:start + 32]])
        started = time.perf_counter()
        probability_batches.append(model.predict(batch, verbose=0))
        inference_seconds += time.perf_counter() - started
    probabilities = np.concatenate(probability_batches, axis=0)
    predictions = probabilities.argmax(axis=1).tolist()
    truth = [label for _, label in examples]
    predicted = [labels[index] for index in predictions]
    confusion = {label: {other: 0 for other in labels} for label in labels}
    for actual, guess in zip(truth, predicted):
        confusion[actual][guess] += 1
    support = Counter(truth)
    predicted_support = Counter(predicted)
    correct = sum(actual == guess for actual, guess in zip(truth, predicted))
    recall = {label: round(confusion[label][label] / support[label], 4) if support[label] else None for label in labels}
    precision = {
        label: round(confusion[label][label] / predicted_support[label], 4) if predicted_support[label] else None
        for label in labels
    }
    f1 = {
        label: round(2 * precision[label] * recall[label] / (precision[label] + recall[label]), 4)
        if precision[label] is not None and recall[label] is not None and precision[label] + recall[label] else None
        for label in labels
    }
    present = [score for score in recall.values() if score is not None]
    present_f1 = [score for score in f1.values() if score is not None]
    n = len(truth)
    p = correct / n
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    confidences = probabilities.max(axis=1)
    correct_flags = np.array([actual == guess for actual, guess in zip(truth, predicted)], dtype=np.float32)
    calibration = []
    expected_calibration_error = 0.0
    for lower in np.arange(0.0, 1.0, 0.2):
        upper = lower + 0.2
        members = (confidences >= lower) & (confidences <= upper if upper >= 1 else confidences < upper)
        count = int(members.sum())
        if not count:
            continue
        bin_accuracy = float(correct_flags[members].mean())
        mean_confidence = float(confidences[members].mean())
        expected_calibration_error += count / n * abs(bin_accuracy - mean_confidence)
        calibration.append({
            "range": [round(float(lower), 1), round(float(min(upper, 1)), 1)],
            "count": count,
            "accuracy": round(bin_accuracy, 4),
            "mean_confidence": round(mean_confidence, 4),
        })
    truth_indices = np.array([labels.index(label) for label in truth])
    one_hot = np.eye(len(labels), dtype=np.float32)[truth_indices]
    mistakes = sorted(
        ({"example_index": index, "actual": actual, "predicted": guess,
          "confidence": round(float(confidences[index]), 4)}
         for index, (actual, guess) in enumerate(zip(truth, predicted)) if actual != guess),
        key=lambda row: row["confidence"], reverse=True,
    )[:10]
    return {
        "correct": correct,
        "total": n,
        "accuracy": round(p, 4),
        "accuracy_95pct_wilson_interval": [round(center - half_width, 4), round(center + half_width, 4)],
        "balanced_accuracy_present_classes": round(float(np.mean(present)), 4),
        "macro_f1_present_classes": round(float(np.mean(present_f1)), 4),
        "uniform_random_expected_accuracy": round(1 / len(labels), 4),
        "majority_class_accuracy": round(max(support.values()) / n, 4),
        "support": {label: support[label] for label in labels},
        "precision_by_class": precision,
        "recall_by_class": recall,
        "f1_by_class": f1,
        "confusion_actual_rows_predicted_columns": confusion,
        "calibration": {
            "expected_calibration_error": round(expected_calibration_error, 4),
            "multiclass_brier_score": round(float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))), 4),
            "bins": calibration,
        },
        "inference_milliseconds_per_image": round(1000 * inference_seconds / n, 3),
        "highest_confidence_mistakes": mistakes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen garment classifier on independent images")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path, help="CSV with image,label; paths relative to the CSV")
    input_group.add_argument("--fashionpedia-images", type=Path, help="Fashionpedia validation parquet")
    parser.add_argument("--fashionpedia-annotations", type=Path, help="Original Fashionpedia validation JSON")
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fashionpedia_images and (not args.fashionpedia_annotations or args.per_class < 1):
        parser.error("Fashionpedia mode needs --fashionpedia-annotations and a positive --per-class.")
    if args.manifest and args.fashionpedia_annotations:
        parser.error("--fashionpedia-annotations only applies to Fashionpedia mode.")

    model, labels = _load_model(str(args.model), str(args.labels), args.model.stat().st_mtime_ns)
    if set(labels) != set(LABELS):
        raise ValueError("Classifier label file does not match the four supported upload categories.")
    if args.manifest:
        examples, metadata = manifest_examples(args.manifest)
        results = {"photos": metrics(model, labels, examples)}
    else:
        context, isolated, metadata = fashionpedia_examples(args.fashionpedia_images, args.fashionpedia_annotations, args.per_class, args.seed)
        results = {"context_crop": metrics(model, labels, context), "isolated_crop": metrics(model, labels, isolated)}
    report = {
        **metadata,
        "classifier_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "labels": labels,
        "results": results,
        "caveat": "This is an evaluation of the frozen classifier; these images were not used to fit it or stop its training.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: {"accuracy": result["accuracy"], "macro_f1": result["macro_f1_present_classes"], "n": result["total"]} for name, result in results.items()}))


if __name__ == "__main__":
    main()
