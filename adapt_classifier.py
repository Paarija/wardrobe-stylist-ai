"""Fine-tune the classifier head using an account's corrected wardrobe photos.

The personal checkpoint stays on this machine under models/users/.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from fashion_ai.classifier import LABELS_PATH, MODEL_PATH, image_array
from fashion_ai.db import list_items
from evaluate_classifier import manifest_examples, metrics
from train_classifier import features_for_images, sample_dataset


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    angle = rng.uniform(-9, 9)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor="white")
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
    return ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt garment classifier to corrected wardrobe photos")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--public-per-class", type=int, default=40)
    parser.add_argument("--variants-per-item", type=int, default=24)
    parser.add_argument("--evaluation-manifest", type=Path, required=True,
                        help="Held-out reviewed photos that are never used for adaptation")
    args = parser.parse_args()

    import tensorflow as tf

    items = [item for item in list_items(args.user_id)
             if item.get("reviewed") == 1 and Path(item["image_path"]).is_file()]
    if not items:
        raise ValueError("The account has no human-reviewed wardrobe labels. Review items in My wardrobe first.")
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    original_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    evaluation_examples, evaluation_metadata = manifest_examples(args.evaluation_manifest)
    base_evaluation = metrics(original_model, labels, evaluation_examples)
    embedder = tf.keras.Model(original_model.input, original_model.layers[-2].output)
    head = original_model.layers[-1]
    rng = random.Random(43)
    images, targets = [], []

    public = sample_dataset(args.dataset, args.public_per_class, seed=43)
    for label, blobs in public.items():
        for blob in blobs:
            try:
                with Image.open(io.BytesIO(blob)) as source:
                    images.append(image_array(source))
                targets.append(labels.index(label))
            except (OSError, ValueError):
                continue

    for item in items:
        with Image.open(item["image_path"]) as source:
            image = source.convert("RGB")
        for variant in range(args.variants_per_item):
            images.append(image_array(image if variant == 0 else augment(image, rng)))
            category = "top" if item["category"] == "outerwear" else item["category"]
            targets.append(labels.index(category))

    features = features_for_images(embedder, images)
    order = np.random.default_rng(43).permutation(len(features))
    head.compile(optimizer=tf.keras.optimizers.Adam(0.0002), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    head.fit(features[order], np.array(targets)[order], batch_size=32, epochs=15, verbose=2)
    personal_model = tf.keras.Model(original_model.input, head(original_model.layers[-2].output))
    output = MODEL_PATH.parent / "users" / f"{args.user_id}.keras"
    output.parent.mkdir(parents=True, exist_ok=True)
    personal_model.save(output)
    adapted_evaluation = metrics(personal_model, labels, evaluation_examples)
    accepted = adapted_evaluation["macro_f1_present_classes"] > base_evaluation["macro_f1_present_classes"]
    evaluation_report = {
        **evaluation_metadata,
        "adaptation_items": len(items),
        "selection_metric": "macro_f1_present_classes",
        "global_model": base_evaluation,
        "adapted_model": adapted_evaluation,
        "accepted": accepted,
        "note": "The held-out evaluation manifest was not used for fitting. The app activates the adapted model only when macro F1 improves.",
    }
    evaluation_path = output.with_suffix(".evaluation.json")
    evaluation_path.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps({
            "base_classifier_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            "accepted": accepted,
            "selection_metric": "macro_f1_present_classes",
            "global_score": base_evaluation["macro_f1_present_classes"],
            "adapted_score": adapted_evaluation["macro_f1_present_classes"],
            "evaluation_report": evaluation_path.name,
        }, indent=2),
        encoding="utf-8",
    )
    saved_features = features[-len(items) * args.variants_per_item::args.variants_per_item]
    truth = np.array([labels.index("top" if item["category"] == "outerwear" else item["category"]) for item in items])
    predicted = head.predict(saved_features, verbose=0).argmax(axis=1)
    print(json.dumps({
        "saved_wardrobe_training_accuracy": float((predicted == truth).mean()),
        "items": len(items),
        "checkpoint": str(output),
        "global_macro_f1": base_evaluation["macro_f1_present_classes"],
        "adapted_macro_f1": adapted_evaluation["macro_f1_present_classes"],
        "accepted": accepted,
    }, indent=2))


if __name__ == "__main__":
    main()
