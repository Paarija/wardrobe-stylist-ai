"""Reproducible small hyperparameter search for ATR outfit segmentation.

Uses the existing Fashionpedia selection split. Neither the previously used
holdout nor a new final holdout is read while choosing a configuration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRIALS = {
    "low_lr": {"learning_rate": 1e-5, "clothing_weight": 1.0, "size": 256, "flip_probability": 0.5, "epochs": 1},
    "mid_lr": {"learning_rate": 3e-5, "clothing_weight": 1.5, "size": 256, "flip_probability": 0.5, "epochs": 1},
    "high_res": {"learning_rate": 1e-5, "clothing_weight": 1.0, "size": 384, "flip_probability": 0.5, "epochs": 1},
    "no_flip": {"learning_rate": 3e-5, "clothing_weight": 1.0, "size": 256, "flip_probability": 0.0, "epochs": 1},
}


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> None:
    tuning_dir = ROOT / "models" / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    for name, params in TRIALS.items():
        output = tuning_dir / name
        report_path = output / "training_report.json"
        if report_path.is_file():
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            report_keys = {"size": "image_size", "epochs": "epochs_requested"}
            if all(saved.get(report_keys.get(key, key)) == value for key, value in params.items()) and (output / "model.safetensors").is_file():
                print(f"Reusing completed trial {name}", flush=True)
                continue
        print(f"Training {name}: {params}", flush=True)
        run(
            "fine_tune_atr.py", "--dataset", "data/training/atr-train-00000.parquet",
            "--train-samples", "192", "--val-samples", "48", "--seed", "42",
            "--output", str(output),
            *(piece for key, value in params.items() for piece in ("--" + key.replace("_", "-"), str(value))),
        )

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    split_report = reports_dir / "segmentation_tuning_validation.json"
    validation_path = reports_dir / "segmentation_tuning_current.json"
    extras = [piece for name in TRIALS for piece in ("--extra-checkpoint", f"{name}=models/tuning/{name}")]
    print("Evaluating all trials on the fixed external selection split", flush=True)
    run(
        "evaluate_fashionpedia.py",
        "--images", "data/training/fashionpedia-val.parquet",
        "--annotations", "data/training/fashionpedia-val-annotations.json",
        "--include-report", str(split_report),
        "--checkpoint", "models/atr-segformer",
        "--base", "models/atr-segformer",
        *extras,
        "--output", str(validation_path),
    )
    evaluation = json.loads(validation_path.read_text(encoding="utf-8"))
    compared_names = {"base", *TRIALS}
    metrics = {name: model["supported_clothing_miou"] for name, model in evaluation["models"].items() if name in compared_names}
    winner = max(metrics, key=lambda name: (metrics[name], name == "base"))
    summary = {
        "chosen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_set": f"Fashionpedia validation: the IDs recorded in {split_report.relative_to(ROOT).as_posix()}",
        "selection_metric": "supported_clothing_miou (top, bottom, shoes, dress)",
        "selection_scores": metrics,
        "trial_hyperparameters": TRIALS,
        "selected_name": winner,
        "selected_checkpoint": Path(evaluation["models"][winner]["checkpoint"]).as_posix(),
        "selected_sha256": evaluation["models"][winner]["checkpoint_sha256"],
        "validation_report": validation_path.relative_to(ROOT).as_posix(),
    }
    summary_path = reports_dir / "segmentation_tuning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"selected": winner, "scores": metrics}), flush=True)


if __name__ == "__main__":
    main()
