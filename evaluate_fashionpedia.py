"""Evaluate outfit parsing on the independent Fashionpedia validation set.

Fashionpedia has instance masks for garments, while this app predicts one
semantic class per pixel. This script merges garment instances into five app
classes and reports pooled pixel IoU for a fixed random sample.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
from PIL import Image
from pycocotools import mask as coco_mask

from fashion_ai.segmentation import LABELS, merge_outerwear_labels, merge_outerwear_probability, predict_class_probability, predict_mask


# Original Fashionpedia category IDs. Garment parts and accessories remain
# background in the app's coarse label scheme.
FASHIONPEDIA_TO_APP = {
    0: 1, 1: 1, 2: 1,              # shirt, top, sweater
    3: 5, 4: 5, 5: 5, 9: 5, 12: 5,  # cardigan, jacket, vest, coat, cape
    6: 2, 7: 2, 8: 2,              # pants, shorts, skirt
    10: 4, 11: 4,                  # dress, jumpsuit
    23: 3,                         # shoe
}
DRAW_PRIORITY = {1: 1, 2: 2, 4: 3, 5: 4, 3: 5}
SUPPORTED = (1, 2, 3, 4)
EVALUATED = (1, 2, 3, 4, 5)


def instance_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        encoded = coco_mask.frPyObjects(segmentation, height, width)
        encoded = coco_mask.merge(encoded)
    elif isinstance(segmentation["counts"], list):
        encoded = coco_mask.frPyObjects(segmentation, height, width)
    else:
        encoded = segmentation
        if isinstance(encoded["counts"], str):
            encoded = {**encoded, "counts": encoded["counts"].encode("ascii")}
    return coco_mask.decode(encoded).astype(bool)


def ground_truth(image: Image.Image, annotations: list[dict]) -> np.ndarray:
    width, height = image.size
    labels = np.zeros((height, width), dtype=np.uint8)
    relevant = [annotation for annotation in annotations if annotation["category_id"] in FASHIONPEDIA_TO_APP]
    relevant.sort(key=lambda annotation: DRAW_PRIORITY[FASHIONPEDIA_TO_APP[annotation["category_id"]]])
    for annotation in relevant:
        labels[instance_mask(annotation, height, width)] = FASHIONPEDIA_TO_APP[annotation["category_id"]]
    return labels


def scores(intersection: np.ndarray, union: np.ndarray) -> dict:
    per_class = {
        LABELS[label_id]: round(float(intersection[label_id] / union[label_id]), 4) if union[label_id] else None
        for label_id in EVALUATED
    }
    supported = [per_class[LABELS[label_id]] for label_id in SUPPORTED if per_class[LABELS[label_id]] is not None]
    all_classes = [score for score in per_class.values() if score is not None]
    return {
        "supported_clothing_miou": round(float(np.mean(supported)), 4),
        "all_clothing_miou": round(float(np.mean(all_classes)), 4),
        "iou_by_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on Fashionpedia validation masks")
    parser.add_argument("--images", type=Path, required=True, help="Fashionpedia validation parquet")
    parser.add_argument("--annotations", type=Path, required=True, help="Original instances_attributes_val2020.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-report", type=Path, action="append", help="Exclude image IDs recorded in an earlier report; repeatable")
    parser.add_argument("--include-report", type=Path, help="Evaluate exactly the image IDs in an earlier report")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/outfit-segformer"))
    parser.add_argument("--base", type=Path, default=Path("models/atr-segformer"))
    parser.add_argument("--extra-checkpoint", action="append", default=[], metavar="NAME=PATH", help="Additional model to compare; repeatable")
    parser.add_argument("--outerwear-overlay", action="append", choices=["top", "background", "top_or_background", "all"], help="Overlay the fine-tuned model's outerwear on base predictions; repeatable")
    parser.add_argument("--outerwear-threshold", type=float, action="append", help="Overlay outerwear on base tops when fine-tuned class probability exceeds this value; repeatable")
    parser.add_argument("--output", type=Path, default=Path("models/outfit-segformer/unseen_eval.json"))
    args = parser.parse_args()

    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))

    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotations_by_image = defaultdict(list)
    for annotation in source["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    rows = parquet.read_table(args.images, columns=["image_id", "image"]).to_pylist()
    if args.exclude_report:
        excluded = set().union(*(set(json.loads(path.read_text(encoding="utf-8"))["image_ids"]) for path in args.exclude_report))
        rows = [row for row in rows if row["image_id"] not in excluded]
    if args.include_report:
        image_ids = json.loads(args.include_report.read_text(encoding="utf-8"))["image_ids"]
        rows_by_id = {row["image_id"]: row for row in rows}
        if not set(image_ids).issubset(rows_by_id):
            parser.error("The include report contains IDs absent from the remaining validation rows.")
        sampled = [rows_by_id[image_id] for image_id in image_ids]
    else:
        if not 1 <= args.limit <= len(rows):
            parser.error(f"--limit must be between 1 and {len(rows)}")
        sampled = random.Random(args.seed).sample(rows, args.limit)
    checkpoints = {"fine_tuned": args.checkpoint, "base": args.base}
    for entry in args.extra_checkpoint:
        if "=" not in entry:
            parser.error("--extra-checkpoint must have the form NAME=PATH")
        name, path = entry.split("=", 1)
        if not name or name in checkpoints:
            parser.error(f"Duplicate or missing checkpoint name: {name!r}")
        checkpoints[name] = Path(path)
    counts = {name: (np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)) for name in checkpoints}
    overlays = args.outerwear_overlay or []
    thresholds = args.outerwear_threshold or []
    if any(not 0 <= value <= 1 for value in thresholds):
        parser.error("Outerwear probability thresholds must lie between 0 and 1.")
    counts.update({f"overlay_{mode}": (np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)) for mode in overlays})
    counts.update({f"threshold_{value:g}": (np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)) for value in thresholds})
    images_with_class = np.zeros(6, dtype=np.int64)

    def add_counts(name: str, truth: np.ndarray, predicted: np.ndarray) -> None:
        intersection, union = counts[name]
        for label_id in EVALUATED:
            target = truth == label_id
            output = predicted == label_id
            intersection[label_id] += np.logical_and(target, output).sum()
            union[label_id] += np.logical_or(target, output).sum()

    for number, row in enumerate(sampled, 1):
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        truth = ground_truth(image, annotations_by_image[row["image_id"]])
        for label_id in EVALUATED:
            images_with_class[label_id] += int(np.any(truth == label_id))
        predicted_cache = {}
        for name, checkpoint in checkpoints.items():
            key = str(checkpoint.resolve())
            if key not in predicted_cache:
                predicted_cache[key] = np.asarray(predict_mask(image, checkpoint))
            predicted = predicted_cache[key]
            add_counts(name, truth, predicted)
        if overlays or thresholds:
            base_mask = predicted_cache[str(args.base.resolve())]
            fine_mask = predicted_cache[str(args.checkpoint.resolve())]
            for mode in overlays:
                combined = merge_outerwear_labels(base_mask, fine_mask, mode)
                add_counts(f"overlay_{mode}", truth, combined)
            if thresholds:
                probability = predict_class_probability(image, args.checkpoint, 5)
                for value in thresholds:
                    combined = merge_outerwear_probability(base_mask, probability, value)
                    add_counts(f"threshold_{value:g}", truth, combined)
        if number % 10 == 0:
            print(f"Evaluated {number}/{len(sampled)} images", flush=True)

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Fashionpedia 2020 validation images and original COCO instance masks",
        "image_count": len(sampled),
        "seed": args.seed,
        "excluded_reports": [str(path) for path in args.exclude_report or []],
        "included_report": str(args.include_report) if args.include_report else None,
        "image_ids": [row["image_id"] for row in sampled],
        "images_with_class": {LABELS[label_id]: int(images_with_class[label_id]) for label_id in EVALUATED},
        "metric": "pooled pixel intersection over union after merging Fashionpedia garment instances",
        "notes": [
            "ATR model was trained on ATR; these Fashionpedia validation images were not used for local fine-tuning.",
            "The base model card lists ATR training data; exact duplicate checking against all original training images was not possible.",
            "Fashionpedia labels and instance masks differ from ATR's semantic labels.",
            "The ATR base checkpoint has no separate outerwear class; an adapted model or ensemble may detect it.",
            "Unannotated pixels are treated as background; Fashionpedia annotations may omit visible garments.",
        ],
        "models": {},
    }
    for name, checkpoint in checkpoints.items():
        weights = checkpoint / "model.safetensors"
        report["models"][name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            **scores(*counts[name]),
        }
    for mode in overlays:
        report["models"][f"overlay_{mode}"] = {
            "checkpoint": f"{args.base} + {args.checkpoint}",
            "base_checkpoint_sha256": report["models"]["base"]["checkpoint_sha256"],
            "outerwear_checkpoint_sha256": report["models"]["fine_tuned"]["checkpoint_sha256"],
            "overlay_mode": mode,
            **scores(*counts[f"overlay_{mode}"]),
        }
    for value in thresholds:
        report["models"][f"threshold_{value:g}"] = {
            "checkpoint": f"{args.base} + {args.checkpoint}",
            "base_checkpoint_sha256": report["models"]["base"]["checkpoint_sha256"],
            "outerwear_checkpoint_sha256": report["models"]["fine_tuned"]["checkpoint_sha256"],
            "outerwear_probability_threshold": value,
            **scores(*counts[f"threshold_{value:g}"]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: model["iou_by_class"] for name, model in report["models"].items()}), flush=True)


if __name__ == "__main__":
    main()
