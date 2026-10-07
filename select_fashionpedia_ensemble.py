"""Select a calibrated outerwear overlay from the fixed Fashionpedia development set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fashion_ai.checkpoints import config_sha256


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "fashionpedia_threshold_refined.json"
BASE = ROOT / "models" / "atr-segformer"
OUTERWEAR = ROOT / "models" / "fashionpedia-segformer"
ENSEMBLE = ROOT / "models" / "fashionpedia-ensemble"
SUMMARY = ROOT / "reports" / "fashionpedia_accuracy_summary.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_threshold(report: dict, base_sha: str, outerwear_sha: str) -> str:
    """Validate report provenance and return the best eligible threshold name."""
    image_ids = report.get("image_ids")
    scores = report.get("models")
    if not isinstance(image_ids, list) or not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("The evaluation report needs a non-empty, duplicate-free image_ids split.")
    if not isinstance(scores, dict) or "base" not in scores or "fine_tuned" not in scores:
        raise ValueError("The evaluation report is missing required model results.")
    if scores["base"].get("checkpoint_sha256") != base_sha:
        raise ValueError("The evaluation report does not match the selected ATR weights.")
    if scores["fine_tuned"].get("checkpoint_sha256") != outerwear_sha:
        raise ValueError("The evaluation report does not match the selected Fashionpedia weights.")
    for name, model in scores.items():
        if not name.startswith("threshold_"):
            continue
        try:
            named_threshold = float(name.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid threshold result name: {name}") from exc
        if (model.get("base_checkpoint_sha256") != base_sha
                or model.get("outerwear_checkpoint_sha256") != outerwear_sha
                or model.get("outerwear_probability_threshold") != named_threshold):
            raise ValueError(f"The {name} result does not match its model weights or threshold.")
    base = scores["base"]
    allowed = {
        name: model for name, model in scores.items()
        if name.startswith("threshold_")
        and model["supported_clothing_miou"] >= base["supported_clothing_miou"] - 0.001
        and model["all_clothing_miou"] > base["all_clothing_miou"]
    }
    if not allowed:
        raise RuntimeError("No threshold improved five-class IoU while preserving the four established classes.")
    return max(allowed, key=lambda name: (allowed[name]["all_clothing_miou"], float(name.split("_", 1)[1])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a verified Fashionpedia outerwear ensemble")
    parser.add_argument("--report", type=Path, default=REPORT, help="Threshold evaluation report to select from")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--outerwear", type=Path, default=OUTERWEAR)
    parser.add_argument("--ensemble", type=Path, default=ENSEMBLE)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scores = report["models"]
    base = scores["base"]
    base_sha = sha256(args.base / "model.safetensors")
    outerwear_sha = sha256(args.outerwear / "model.safetensors")
    chosen = select_threshold(report, base_sha, outerwear_sha)
    threshold = float(chosen.split("_", 1)[1])
    config = {
        "base": Path(os.path.relpath(args.base.resolve(), args.ensemble.resolve())).as_posix(),
        "outerwear": Path(os.path.relpath(args.outerwear.resolve(), args.ensemble.resolve())).as_posix(),
        "threshold": threshold,
    }
    args.ensemble.mkdir(parents=True, exist_ok=True)
    config_path = args.ensemble / "ensemble.json"
    with config_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
    summary = {
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_report": report_path.relative_to(ROOT).as_posix() if report_path.is_relative_to(ROOT) else str(report_path),
        "selection_images": report["image_ids"],
        "selection_rule": "Highest five-class mean IoU among thresholds with four-class mean IoU no more than 0.001 below the original model",
        "selected_name": chosen,
        "selected_checkpoint": args.ensemble.resolve().relative_to(ROOT).as_posix(),
        "selected_sha256": config_sha256(config),
        "components": {
            "base": {"checkpoint": args.base.resolve().relative_to(ROOT).as_posix(), "sha256": base_sha},
            "outerwear": {"checkpoint": args.outerwear.resolve().relative_to(ROOT).as_posix(), "sha256": outerwear_sha},
        },
        "base_scores": {key: base[key] for key in ("supported_clothing_miou", "all_clothing_miou", "iou_by_class")},
        "selected_scores": {key: scores[chosen][key] for key in ("supported_clothing_miou", "all_clothing_miou", "iou_by_class")},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"selected": chosen, "base_five_class_miou": base["all_clothing_miou"], "selected_five_class_miou": scores[chosen]["all_clothing_miou"]}))


if __name__ == "__main__":
    main()
