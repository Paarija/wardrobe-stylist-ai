"""Select only segmentation checkpoints whose bundled weights match their reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_sha256(config: dict) -> str:
    """Hash JSON values instead of OS-dependent line endings and whitespace."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def has_label_count(path: Path, expected: int) -> bool:
    config = json.loads(path.read_text(encoding="utf-8"))
    return len(config.get("id2label", {})) == expected


def active_segmentation_checkpoint(root: Path) -> Path:
    models = root / "models"
    base = models / "atr-segformer"
    base_weights = base / "model.safetensors"
    tuning_report = root / "reports" / "segmentation_tuning_summary.json"
    if not tuning_report.is_file() or not base_weights.is_file() or not (base / "config.json").is_file():
        raise RuntimeError("The bundled ATR segmentation checkpoint or its verification report is missing.")
    if not has_label_count(base / "config.json", 18):
        raise RuntimeError("The bundled ATR segmentation configuration must have 18 labels.")
    tuned = json.loads(tuning_report.read_text(encoding="utf-8"))
    if tuned.get("selected_checkpoint") != "models/atr-segformer" or sha256(base_weights) != tuned.get("selected_sha256"):
        raise RuntimeError("The bundled ATR segmentation weights do not match the verification report.")

    selection_path = root / "reports" / "fashionpedia_accuracy_summary.json"
    ensemble = models / "fashionpedia-ensemble"
    config_path = ensemble / "ensemble.json"
    outerwear = models / "fashionpedia-segformer"
    outerwear_weights = outerwear / "model.safetensors"
    if not (selection_path.is_file() and config_path.is_file() and outerwear_weights.is_file()
            and (outerwear / "config.json").is_file()):
        return base
    try:
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        components = selected.get("components", {})
        if (selected.get("selected_checkpoint") == "models/fashionpedia-ensemble"
                and selected.get("selected_sha256") == config_sha256(config)
                and config.get("base") == "../atr-segformer"
                and config.get("outerwear") == "../fashionpedia-segformer"
                and isinstance(config.get("threshold"), (int, float))
                and 0 <= config["threshold"] <= 1
                and components.get("base", {}).get("checkpoint") == "models/atr-segformer"
                and components.get("base", {}).get("sha256") == tuned["selected_sha256"]
                and components.get("outerwear", {}).get("checkpoint") == "models/fashionpedia-segformer"
                and components.get("outerwear", {}).get("sha256") == sha256(outerwear_weights)
                and has_label_count(outerwear / "config.json", 6)):
            return ensemble
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        # The ensemble is optional. Any malformed optional metadata falls back
        # to the already verified ATR checkpoint.
        return base
    return base
