"""Fine-tune the small ATR clothes parser on local image/mask pairs.

The supplied ATR base checkpoint was trained on the same public dataset. The
validation split here tracks this local fine-tuning run; it is not an unseen
test set for the base checkpoint.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
from PIL import Image

from fashion_ai.segmentation import ATR_TO_APP, LABELS, MEAN, STD


ATR_LEFT_RIGHT_PAIRS = ((9, 10), (12, 13), (14, 15))


def swap_atr_left_right(labels: np.ndarray) -> np.ndarray:
    """Swap every directional ATR class after a horizontal image flip."""
    swapped = labels.copy()
    for left_id, right_id in ATR_LEFT_RIGHT_PAIRS:
        swapped[labels == left_id] = right_id
        swapped[labels == right_id] = left_id
    return swapped


def read_pairs(path: Path, count: int) -> list[tuple[bytes, bytes]]:
    pairs = []
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(batch_size=64, columns=["image", "mask"]):
        for row in batch.to_pylist():
            pairs.append((row["image"]["bytes"], row["mask"]["bytes"]))
            if len(pairs) == count:
                return pairs
    raise ValueError(f"The parquet has only {len(pairs)} image/mask pairs; requested {count}.")


def iou_report(model, loader, device) -> dict:
    import torch
    import torch.nn.functional as functional

    intersection = np.zeros(len(LABELS), dtype=np.int64)
    union = np.zeros(len(LABELS), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for pixels, raw_labels in loader:
            logits = model(pixel_values=pixels.to(device)).logits
            logits = functional.interpolate(logits, size=raw_labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = ATR_TO_APP[logits.argmax(dim=1).cpu().numpy()]
            truth = ATR_TO_APP[raw_labels.numpy()]
            for label_id in LABELS:
                intersection[label_id] += np.logical_and(predicted == label_id, truth == label_id).sum()
                union[label_id] += np.logical_or(predicted == label_id, truth == label_id).sum()
    per_class = {
        LABELS[index]: round(float(intersection[index] / union[index]), 4) if union[index] else None
        for index in LABELS
    }
    clothing = [score for name, score in per_class.items() if name != "background" and score is not None]
    return {"clothing_miou": round(float(np.mean(clothing)), 4), "iou_by_class": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the ATR clothes parser")
    parser.add_argument("--dataset", type=Path, required=True, help="ATR image/mask parquet shard")
    parser.add_argument("--base", type=Path, default=Path("models/atr-segformer"))
    parser.add_argument("--output", type=Path, default=Path("models/atr-finetuned"))
    parser.add_argument("--train-samples", type=int, default=192)
    parser.add_argument("--val-samples", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--clothing-weight", type=float, default=1.5)
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_samples < 1 or args.val_samples < 1 or args.size < 64 or args.learning_rate <= 0 or args.clothing_weight <= 0 or not 0 <= args.flip_probability <= 1:
        parser.error("Use positive sample counts, image size, learning rate and class weight; flip probability must be 0 to 1.")

    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, Dataset
    from transformers import SegformerForSemanticSegmentation

    class ATRDataset(Dataset):
        def __init__(self, pairs, augment=False):
            self.pairs = pairs
            self.augment = augment

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, index):
            image_bytes, mask_bytes = self.pairs[index]
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB").resize((args.size, args.size), Image.Resampling.BILINEAR)
            with Image.open(io.BytesIO(mask_bytes)) as source:
                mask = source.convert("L").resize((args.size, args.size), Image.Resampling.NEAREST)
            flipped = self.augment and random.random() < args.flip_probability
            if flipped:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            pixels = ((pixels - MEAN) / STD).transpose(2, 0, 1).copy()
            labels = np.asarray(mask, dtype=np.int64).copy()
            if labels.max() >= len(ATR_TO_APP):
                raise ValueError("ATR mask contains an unknown class ID.")
            if flipped:
                labels = swap_atr_left_right(labels)
            return torch.from_numpy(pixels), torch.from_numpy(labels)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    pairs = read_pairs(args.dataset, args.train_samples + args.val_samples)
    random.shuffle(pairs)
    train_loader = DataLoader(ATRDataset(pairs[:args.train_samples], True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ATRDataset(pairs[args.train_samples:]), batch_size=args.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(args.base, local_files_only=True).to(device)
    if model.config.num_labels != len(ATR_TO_APP):
        raise ValueError("The base checkpoint must use the 18 ATR labels.")
    for parameter in model.segformer.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.decode_head.parameters(), lr=args.learning_rate)
    weights = torch.ones(len(ATR_TO_APP), device=device)
    weights[[4, 5, 6, 7, 9, 10]] = args.clothing_weight
    baseline = iou_report(model, val_loader, device)
    print(json.dumps({"stage": "base", **baseline}), flush=True)
    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for pixels, labels in train_loader:
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=pixels).logits
            logits = functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = functional.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        metrics = iou_report(model, val_loader, device)
        report = {"epoch": epoch, "training_loss": round(total_loss / len(train_loader), 4), **metrics}
        print(json.dumps(report), flush=True)
        if metrics["clothing_miou"] > best_miou:
            best_miou = metrics["clothing_miou"]
            args.output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output)
            (args.output / "training_report.json").write_text(json.dumps({
                "source": "mattmdjaga/human_parsing_dataset (ATR), first parquet shard",
                "base_checkpoint": "mattmdjaga/segformer_b0_clothes",
                "train_samples": args.train_samples,
                "validation_samples": args.val_samples,
                "validation_unseen_by_base_model": False,
                "image_size": args.size,
                "learning_rate": args.learning_rate,
                "clothing_weight": args.clothing_weight,
                "flip_probability": args.flip_probability,
                "seed": args.seed,
                "epochs_requested": args.epochs,
                "device": device,
                "baseline": baseline,
                "best": report,
            }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
