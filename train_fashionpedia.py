"""Adapt the ATR SegFormer to the app's six labels using Fashionpedia train masks.

Uses only the official Fashionpedia training split. The official validation
images are never read by this script, so they remain available for comparison.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
import pyarrow.parquet as parquet
from PIL import Image

from evaluate_fashionpedia import FASHIONPEDIA_TO_APP, ground_truth
from fashion_ai.segmentation import ATR_TO_APP, LABELS, MEAN, STD


def image_rows(path: Path, count: int, seed: int) -> list[dict]:
    source = parquet.ParquetFile(path)
    image_ids = source.read(columns=["image_id"])["image_id"].to_pylist()
    if count > len(image_ids):
        raise ValueError(f"Requested {count} images, but the parquet has {len(image_ids)}.")
    chosen = set(random.Random(seed).sample(image_ids, count))
    rows = {}
    for batch in source.iter_batches(batch_size=32, columns=["image_id", "image"]):
        for row in batch.to_pylist():
            if row["image_id"] in chosen:
                rows[row["image_id"]] = row
        if len(rows) == count:
            break
    return [rows[image_id] for image_id in sorted(chosen)]


def chosen_annotations(path: Path, ids: set[int], cache: Path) -> dict[int, list[dict]]:
    if cache.is_file():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if set(saved["image_ids"]) == ids:
            return {int(key): value for key, value in saved["annotations"].items()}
    selected = defaultdict(list)
    with path.open("rb") as stream:
        for annotation in ijson.items(stream, "annotations.item", use_float=True):
            if annotation["image_id"] in ids and annotation["category_id"] in FASHIONPEDIA_TO_APP:
                selected[annotation["image_id"]].append(annotation)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"image_ids": sorted(ids), "annotations": selected}), encoding="utf-8")
    return selected


def prepared_pairs(rows: list[dict], annotations: dict[int, list[dict]], size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for index, row in enumerate(rows, 1):
        with Image.open(io.BytesIO(row["image"]["bytes"])) as source:
            image = source.convert("RGB")
        mask = ground_truth(image, annotations.get(row["image_id"], []))
        pixels = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8).copy()
        labels = np.asarray(Image.fromarray(mask).resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8).copy()
        pairs.append((pixels, labels))
        if index % 100 == 0:
            print(f"Prepared {index}/{len(rows)} image and mask pairs", flush=True)
    return pairs


def model_from_atr(base_path: Path):
    import torch
    from transformers import SegformerForSemanticSegmentation

    original = SegformerForSemanticSegmentation.from_pretrained(base_path, local_files_only=True)
    if original.config.num_labels != len(ATR_TO_APP):
        raise ValueError("The starting checkpoint must have the 18 ATR labels.")
    config = copy.deepcopy(original.config)
    config.id2label = LABELS.copy()
    config.label2id = {name: index for index, name in LABELS.items()}
    adapted = SegformerForSemanticSegmentation(config)
    transferred = adapted.state_dict()
    for name, weights in original.state_dict().items():
        if name in transferred and transferred[name].shape == weights.shape:
            transferred[name] = weights
    adapted.load_state_dict(transferred)
    # Start the new classes from the related ATR classifier channels.
    groups = {0: [0, 1, 2, 3, 8, 11, 12, 13, 14, 15, 16, 17],
              1: [4], 2: [5, 6], 3: [9, 10], 4: [7], 5: [4]}
    with torch.no_grad():
        for target, sources in groups.items():
            adapted.decode_head.classifier.weight[target] = original.decode_head.classifier.weight[sources].mean(0)
            adapted.decode_head.classifier.bias[target] = original.decode_head.classifier.bias[sources].mean(0)
    return adapted


def iou_report(model, loader, device) -> dict:
    import torch
    import torch.nn.functional as functional

    intersection = np.zeros(len(LABELS), dtype=np.int64)
    union = np.zeros(len(LABELS), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for pixels, labels in loader:
            logits = model(pixel_values=pixels.to(device)).logits
            logits = functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = logits.argmax(dim=1).cpu().numpy()
            truth = labels.numpy()
            for class_id in LABELS:
                intersection[class_id] += np.logical_and(predicted == class_id, truth == class_id).sum()
                union[class_id] += np.logical_or(predicted == class_id, truth == class_id).sum()
    by_class = {LABELS[index]: round(float(intersection[index] / union[index]), 4) if union[index] else None for index in LABELS}
    present = [score for index, score in enumerate(by_class.values()) if index and score is not None]
    return {"clothing_miou": round(float(np.mean(present)), 4), "iou_by_class": by_class}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer on Fashionpedia garment masks")
    parser.add_argument("--images", type=Path, default=Path("data/training/fashionpedia-train-00000.parquet"))
    parser.add_argument("--annotations", type=Path, default=Path("data/training/fashionpedia-train-annotations.json"))
    parser.add_argument("--base", type=Path, default=Path("models/atr-segformer"))
    parser.add_argument("--resume", type=Path, help="Continue from an existing six-label Fashionpedia checkpoint")
    parser.add_argument("--output", type=Path, default=Path("models/fashionpedia-segformer"))
    parser.add_argument("--train-samples", type=int, default=640)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--class-weights", type=float, nargs=6, default=[0.5, 1.0, 1.0, 2.0, 1.0, 1.5], metavar="WEIGHT")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_samples, args.val_samples, args.epochs, args.batch_size, args.size) <= 0 or args.learning_rate <= 0 or min(args.class_weights) <= 0:
        parser.error("Sample counts, epochs, batch size, image size, and learning rate must be positive.")

    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, Dataset
    from transformers import SegformerForSemanticSegmentation

    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = image_rows(args.images, args.train_samples + args.val_samples, args.seed)
    ids = {row["image_id"] for row in rows}
    cache = Path("data/training/fashionpedia-selected-annotations.json")
    annotations = chosen_annotations(args.annotations, ids, cache)
    random.Random(args.seed).shuffle(rows)
    pairs = prepared_pairs(rows, annotations, args.size)

    class Pairs(Dataset):
        def __init__(self, data, augment):
            self.data = data
            self.augment = augment

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            image, mask = self.data[index]
            if self.augment and random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            pixels = ((image.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()
            return torch.from_numpy(pixels), torch.from_numpy(mask.astype(np.int64).copy())

    train_loader = DataLoader(Pairs(pairs[:args.train_samples], True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Pairs(pairs[args.train_samples:], False), batch_size=args.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (SegformerForSemanticSegmentation.from_pretrained(args.resume, local_files_only=True) if args.resume else model_from_atr(args.base)).to(device)
    if model.config.num_labels != len(LABELS):
        raise ValueError("A resumed checkpoint must use the app's six labels.")
    for parameter in model.segformer.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.decode_head.parameters(), lr=args.learning_rate)
    # Rare garments need a larger loss contribution than background pixels.
    class_weights = torch.tensor(args.class_weights, device=device)
    history = []
    best = -1.0
    initial = iou_report(model, val_loader, device)
    print(json.dumps({"stage": "initialized", **initial}), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for pixels, labels in train_loader:
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=pixels).logits
            logits = functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = functional.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        metrics = iou_report(model, val_loader, device)
        entry = {"epoch": epoch, "training_loss": round(loss_sum / len(train_loader), 4), **metrics}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if metrics["clothing_miou"] > best:
            best = metrics["clothing_miou"]
            args.output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output)
            (args.output / "training_report.json").write_text(json.dumps({
                "dataset": "Fashionpedia 2020 official training split, first parquet shard",
                "base_checkpoint": str(args.base),
                "resumed_checkpoint": str(args.resume) if args.resume else None,
                "train_image_ids": [row["image_id"] for row in rows[:args.train_samples]],
                "validation_image_ids": [row["image_id"] for row in rows[args.train_samples:]],
                "train_samples": args.train_samples,
                "validation_samples": args.val_samples,
                "epochs_requested": args.epochs,
                "image_size": args.size,
                "learning_rate": args.learning_rate,
                "class_weights": class_weights.tolist(),
                "seed": args.seed,
                "device": device,
                "initialized": initial,
                "history": history,
                "best": entry,
            }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
