"""LIP label mapping and SegFormer inference for a single outfit photo."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


LABELS = {0: "background", 1: "top", 2: "bottom", 3: "shoes", 4: "dress", 5: "outerwear"}
PALETTE = {1: (85, 164, 255), 2: (255, 176, 86), 3: (156, 216, 118), 4: (204, 139, 241), 5: (250, 113, 127)}
IMAGE_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Official LIP IDs: 5 upper-clothes, 6 dress, 7 coat, 9 pants,
# 10 jumpsuits, 12 skirt, 18/19 left/right shoe. 255 is ignored.
LIP_TO_APP = np.zeros(256, dtype=np.uint8)
LIP_TO_APP[5] = 1
LIP_TO_APP[9] = 2
LIP_TO_APP[12] = 2
LIP_TO_APP[18] = 3
LIP_TO_APP[19] = 3
LIP_TO_APP[6] = 4
LIP_TO_APP[10] = 4
LIP_TO_APP[7] = 5
LIP_TO_APP[255] = 255

# ATR human-parsing IDs used by mattmdjaga/segformer_b0_clothes.
ATR_TO_APP = np.zeros(18, dtype=np.uint8)
ATR_TO_APP[4] = 1  # upper clothes
ATR_TO_APP[5] = 2  # skirt
ATR_TO_APP[6] = 2  # pants
ATR_TO_APP[9] = 3  # left shoe
ATR_TO_APP[10] = 3  # right shoe
ATR_TO_APP[7] = 4  # dress


def remap_lip_mask(mask: Image.Image) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("LIP mask must be a single-channel PNG with class IDs.")
    return LIP_TO_APP[values.astype(np.uint8)]


def remap_atr_mask(mask: Image.Image) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2 or np.any(values >= len(ATR_TO_APP)):
        raise ValueError("ATR mask must contain single-channel class IDs from 0 to 17.")
    return ATR_TO_APP[values.astype(np.uint8)]


def image_tensor_array(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return ((pixels - MEAN) / STD).transpose(2, 0, 1).copy()


@lru_cache(maxsize=8)
def _load_model(checkpoint: str, weights_mtime_ns: int):
    import torch
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, local_files_only=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), device


def predict_mask(image: Image.Image, checkpoint: str | Path) -> Image.Image:
    import torch
    import torch.nn.functional as functional

    checkpoint = Path(checkpoint)
    ensemble_config = checkpoint / "ensemble.json"
    if ensemble_config.is_file():
        config = json.loads(ensemble_config.read_text(encoding="utf-8"))
        base = np.asarray(predict_mask(image, checkpoint / config["base"]))
        if "threshold" in config:
            probability = predict_class_probability(image, checkpoint / config["outerwear"], 5)
            return Image.fromarray(merge_outerwear_probability(base, probability, config["threshold"]))
        outerwear = np.asarray(predict_mask(image, checkpoint / config["outerwear"]))
        return Image.fromarray(merge_outerwear_labels(base, outerwear, config["mode"]))
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        weights = checkpoint / "pytorch_model.bin"
    model, device = _load_model(str(checkpoint), weights.stat().st_mtime_ns)
    pixel_values = torch.from_numpy(image_tensor_array(image)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(pixel_values=pixel_values).logits
        logits = functional.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    if model.config.num_labels == len(ATR_TO_APP):
        labels = ATR_TO_APP[labels]
    elif model.config.num_labels != len(LABELS):
        raise ValueError(f"Unsupported segmentation model with {model.config.num_labels} labels.")
    return Image.fromarray(labels).resize(image.size, Image.Resampling.NEAREST)


def merge_outerwear_labels(base: np.ndarray, outerwear: np.ndarray, mode: str) -> np.ndarray:
    """Keep the base parser's classes and add outerwear from a second parser."""
    if base.shape != outerwear.shape:
        raise ValueError("Segmentation masks must have the same dimensions.")
    if mode == "top":
        allowed = base == 1
    elif mode == "background":
        allowed = base == 0
    elif mode == "top_or_background":
        allowed = (base == 0) | (base == 1)
    elif mode == "all":
        allowed = np.ones(base.shape, dtype=bool)
    else:
        raise ValueError(f"Unknown outerwear merge mode: {mode}")
    combined = base.copy()
    combined[(outerwear == 5) & allowed] = 5
    return combined


def merge_outerwear_probability(base: np.ndarray, probability: np.ndarray, threshold: float) -> np.ndarray:
    if base.shape != probability.shape or not 0 <= threshold <= 1:
        raise ValueError("Use matching mask dimensions and a threshold from 0 to 1.")
    combined = base.copy()
    combined[(base == 1) & (probability >= threshold)] = 5
    return combined


def predict_class_probability(image: Image.Image, checkpoint: str | Path, class_id: int) -> np.ndarray:
    """Return one class's softmax probability at the original image size."""
    import torch
    import torch.nn.functional as functional

    checkpoint = Path(checkpoint)
    weights = checkpoint / "model.safetensors"
    model, device = _load_model(str(checkpoint), weights.stat().st_mtime_ns)
    if not 0 <= class_id < model.config.num_labels:
        raise ValueError(f"Class {class_id} is absent from this checkpoint.")
    pixel_values = torch.from_numpy(image_tensor_array(image)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(pixel_values=pixel_values).logits
        logits = functional.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        probability = logits.softmax(dim=1)[0, class_id].cpu().numpy()
    return np.asarray(Image.fromarray(probability.astype(np.float32)).resize(image.size, Image.Resampling.BILINEAR))


def overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    labels = np.asarray(mask.resize(image.size, Image.Resampling.NEAREST))
    for label_id, color in PALETTE.items():
        selected = labels == label_id
        if selected.any():
            base[selected] = (base[selected].astype(np.float32) * 0.45 + np.array(color) * 0.55).astype(np.uint8)
    return Image.fromarray(base)


def extract_items(image: Image.Image, mask: Image.Image, minimum_pixels: int = 250) -> dict[str, Image.Image]:
    image = image.convert("RGB")
    labels = np.asarray(mask.resize(image.size, Image.Resampling.NEAREST))
    pixels = np.asarray(image).copy()
    results = {}
    for label_id, name in LABELS.items():
        if label_id == 0:
            continue
        selected = labels == label_id
        if int(selected.sum()) < minimum_pixels:
            continue
        ys, xs = np.where(selected)
        isolated = np.full_like(pixels, 255)
        isolated[selected] = pixels[selected]
        crop = Image.fromarray(isolated).crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        results[name] = crop
    return results
