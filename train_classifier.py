"""Train a neural garment classifier on labelled fashion product photos.

Download the Fashion Product Images (Small) parquet described in README first.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from fashion_ai.classifier import IMAGE_SIZE, category_from_dataset, image_array


def sample_dataset(parquet_path: Path, per_class: int, seed: int) -> dict[str, list[bytes]]:
    import pyarrow.parquet as parquet

    rng = random.Random(seed)
    buckets: dict[str, list[tuple[int, str, bytes]]] = defaultdict(list)
    source = parquet.ParquetFile(parquet_path)
    for batch in source.iter_batches(batch_size=256, columns=["id", "masterCategory", "subCategory", "articleType", "image"]):
        for row in batch.to_pylist():
            label = category_from_dataset(row["masterCategory"], row["subCategory"], row["articleType"])
            blob = row["image"]["bytes"] if row["image"] else None
            if label is None or not blob:
                continue
            buckets[label].append((row["id"], row["articleType"], blob))
    result = {}
    for label, records in buckets.items():
        # Product catalogues contain far fewer skirts and women's tops than jeans
        # or T-shirts. Keep examples of both before filling the rest randomly.
        priority_type = {"bottom": "Skirts", "top": "Tops"}.get(label)
        priority = [record for record in records if record[1] == priority_type] if priority_type else []
        priority = rng.sample(priority, min(len(priority), per_class // 3))
        priority_ids = {record[0] for record in priority}
        remaining = [record for record in records if record[0] not in priority_ids]
        chosen = priority + rng.sample(remaining, min(len(remaining), per_class - len(priority)))
        result[label] = [record[2] for record in chosen]
    return result


def features_for_images(model, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    results = []
    for start in range(0, len(images), batch_size):
        batch = np.stack(images[start:start + batch_size])
        results.append(model.predict(batch, verbose=0))
    return np.concatenate(results, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train automatic wardrobe image classification")
    parser.add_argument("--dataset", type=Path, required=True, help="Fashion Product Images Small parquet file")
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--per-class", type=int, default=260)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import tensorflow as tf

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    samples = sample_dataset(args.dataset, args.per_class, args.seed)
    labels = [label for label in ("top", "bottom", "shoes", "dress") if label in samples]
    if set(labels) != {"top", "bottom", "shoes", "dress"}:
        raise ValueError(f"Dataset is missing a category. Found: {labels}")
    print("sampled images:", {label: len(samples[label]) for label in labels}, flush=True)
    train_images, train_targets, val_images, val_targets = [], [], [], []
    for index, label in enumerate(labels):
        decoded = []
        for blob in samples[label]:
            try:
                with Image.open(io.BytesIO(blob)) as source:
                    decoded.append(image_array(source))
            except (OSError, ValueError):
                continue
        if len(decoded) < 50:
            raise ValueError(f"Only {len(decoded)} readable images for {label}.")
        rng = np.random.default_rng(args.seed + index)
        rng.shuffle(decoded)
        boundary = max(1, int(len(decoded) * 0.8))
        train_images.extend(decoded[:boundary])
        train_targets.extend([index] * boundary)
        val_images.extend(decoded[boundary:])
        val_targets.extend([index] * (len(decoded) - boundary))

    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base.trainable = False
    train_features = features_for_images(base, train_images)
    val_features = features_for_images(base, val_images)
    head = tf.keras.Sequential([
        tf.keras.Input(shape=(train_features.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])
    head.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    head.fit(
        train_features, np.array(train_targets), validation_data=(val_features, np.array(val_targets)),
        epochs=args.epochs, batch_size=32, callbacks=[early_stop], verbose=2,
    )
    probabilities = head.predict(val_features, verbose=0)
    predicted = probabilities.argmax(axis=1)
    truth = np.array(val_targets)
    report = {
        "validation_accuracy": round(float((predicted == truth).mean()), 4),
        "per_class_accuracy": {label: round(float((predicted[truth == index] == index).mean()), 4) for index, label in enumerate(labels)},
        "train_examples": len(train_images),
        "validation_examples": len(val_images),
        "source": "Fashion Product Images (Small), via Transformersx Hugging Face mirror",
        "backbone": "MobileNetV2 with ImageNet initialization; frozen during head training",
        "head": "Dense(128) + Dropout(0.25) + four-class softmax",
        "label_scheme": "all_upper_body_garments_including_outerwear_are_tops",
        "seed": args.seed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    combined = tf.keras.Model(base.input, head(base.output), name="garment_classifier")
    combined.save(args.output / "garment_classifier.keras")
    (args.output / "garment_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (args.output / "garment_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
