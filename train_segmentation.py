"""Fine-tune SegFormer-B0 on the LIP outfit parsing masks.

Example:
python train_segmentation.py --train-images path/to/train_images \
  --train-masks path/to/train_segmentations --val-images path/to/val_images \
  --val-masks path/to/val_segmentations --output models/lip-segformer
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from fashion_ai.segmentation import IMAGE_SIZE, LABELS, image_tensor_array, remap_lip_mask


def paired_files(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    images = {path.stem: path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    pairs = [(image, mask_dir / f"{stem}.png") for stem, image in images.items() if (mask_dir / f"{stem}.png").is_file()]
    if not pairs:
        raise ValueError(f"No matching image and PNG mask names in {image_dir} and {mask_dir}.")
    return sorted(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune outfit segmentation on LIP masks")
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-masks", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-masks", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("models/atr-segformer"))
    parser.add_argument("--output", type=Path, default=Path("models/lip-segformer"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, Dataset
    from transformers import SegformerForSemanticSegmentation

    class LIPDataset(Dataset):
        def __init__(self, pairs: list[tuple[Path, Path]], augment: bool):
            self.pairs = pairs
            self.augment = augment

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, index):
            image_path, mask_path = self.pairs[index]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            with Image.open(mask_path) as source:
                raw_mask = source.copy()
            if self.augment and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                raw_mask = raw_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = raw_mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
            pixels = torch.from_numpy(image_tensor_array(image))
            labels = torch.from_numpy(remap_lip_mask(mask).astype(np.int64))
            return pixels, labels

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train_pairs = paired_files(args.train_images, args.train_masks)
    val_pairs = paired_files(args.val_images, args.val_masks)
    train_loader = DataLoader(LIPDataset(train_pairs, True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(LIPDataset(val_pairs, False), batch_size=args.batch_size, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.base,
        num_labels=len(LABELS),
        id2label=LABELS,
        label2id={name: index for index, name in LABELS.items()},
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    class_weights = torch.tensor([0.25, 1.0, 1.0, 2.0, 1.0, 1.0], device=device)
    best_miou = -1.0
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for pixels, labels in train_loader:
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=pixels).logits
            logits = functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = functional.cross_entropy(logits, labels, weight=class_weights, ignore_index=255)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        intersection = np.zeros(len(LABELS), dtype=np.int64)
        union = np.zeros(len(LABELS), dtype=np.int64)
        with torch.inference_mode():
            for pixels, labels in val_loader:
                pixels = pixels.to(device)
                logits = model(pixel_values=pixels).logits
                logits = functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = logits.argmax(dim=1).cpu().numpy()
                truth = labels.numpy()
                valid = truth != 255
                for label_id in LABELS:
                    intersection[label_id] += np.logical_and(predicted == label_id, truth == label_id).sum()
                    union[label_id] += np.logical_and(np.logical_or(predicted == label_id, truth == label_id), valid).sum()
        ious = {LABELS[index]: round(float(intersection[index] / union[index]), 4) if union[index] else None
                for index in LABELS}
        clothing_ious = [ious[LABELS[index]] for index in range(1, len(LABELS)) if ious[LABELS[index]] is not None]
        miou = float(np.mean(clothing_ious)) if clothing_ious else 0.0
        report = {"epoch": epoch, "training_loss": round(train_loss / len(train_loader), 4), "clothing_miou": round(miou, 4), "iou_by_class": ious}
        print(json.dumps(report), flush=True)
        if miou > best_miou:
            best_miou = miou
            model.save_pretrained(args.output)
            (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
