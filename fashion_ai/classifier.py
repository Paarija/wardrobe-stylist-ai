"""Automatic garment category and colour for single-item uploads."""

from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from fashion_ai.color import infer_color as infer_color, infer_colors as infer_colors


IMAGE_SIZE = 160
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "garment_classifier.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "garment_labels.json"


def category_from_dataset(master: str, subcategory: str, article_type: str) -> str | None:
    """Map product labels into four upload categories; outerwear is a top."""
    if master == "Footwear":
        return "shoes"
    if master != "Apparel":
        return None
    if subcategory == "Bottomwear":
        return "bottom"
    if subcategory == "Dress":
        return "dress"
    if subcategory == "Topwear":
        if article_type in {
            "Tshirts", "Shirts", "Kurtas", "Tops", "Tunics", "Kurtis",
            "Sweaters", "Sweatshirts", "Jackets", "Waistcoat", "Blazers",
            "Shrug", "Nehru Jackets", "Rain Jacket",
        }:
            return "top"
    return None


def image_array(image: Image.Image) -> np.ndarray:
    prepared = ImageOps.pad(image.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), color="white", method=Image.Resampling.BILINEAR)
    return np.asarray(prepared, dtype=np.float32) / 127.5 - 1.0


@lru_cache(maxsize=4)
def _load_model(model_path: str, labels_path: str, checkpoint_version: int):
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    return model, labels


def classify_item(
    image: Image.Image,
    model_path: Path = MODEL_PATH,
    labels_path: Path = LABELS_PATH,
    user_id: int | None = None,
) -> tuple[str, float]:
    if not model_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Required garment classifier files are missing. Restore the bundled model and labels.")
    if user_id is not None:
        personal_path = model_path.parent / "users" / f"{user_id}.keras"
        metadata_path = personal_path.with_suffix(".json")
        if personal_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            base_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if metadata.get("base_classifier_sha256") == base_sha and metadata.get("accepted") is True:
                model_path = personal_path
    model, labels = _load_model(str(model_path), str(labels_path), model_path.stat().st_mtime_ns)
    probabilities = model.predict(image_array(image)[None, ...], verbose=0)[0]
    index = int(np.argmax(probabilities))
    return labels[index], float(probabilities[index])
