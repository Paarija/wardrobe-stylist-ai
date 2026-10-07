"""Load each shipped neural model in an isolated process and run inference."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def check_classifier() -> dict:
    from fashion_ai.classifier import classify_item

    image = Image.new("RGB", (64, 96), "white")
    started = time.perf_counter()
    category, confidence = classify_item(image)
    elapsed = (time.perf_counter() - started) * 1000
    if category not in {"top", "bottom", "shoes", "dress"}:
        raise RuntimeError("The classifier returned an invalid category.")
    return {
        "classifier_category": category,
        "classifier_confidence": round(confidence, 4),
        "classifier_cold_start_milliseconds": round(elapsed, 1),
    }


def check_segmentation() -> dict:
    from fashion_ai.checkpoints import active_segmentation_checkpoint
    from fashion_ai.segmentation import predict_mask

    image = Image.new("RGB", (64, 96), "white")
    checkpoint = active_segmentation_checkpoint(ROOT)
    started = time.perf_counter()
    mask = predict_mask(image, checkpoint)
    elapsed = (time.perf_counter() - started) * 1000
    if mask.size != image.size:
        raise RuntimeError("The segmentation model returned an invalid mask size.")
    return {
        "segmentation_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "segmentation_mask_size": mask.size,
        "segmentation_labels": sorted(set(mask.getdata())),
        "segmentation_cold_start_milliseconds": round(elapsed, 1),
    }


def run_isolated(mode: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--classifier", action="store_true")
    group.add_argument("--segmentation", action="store_true")
    args = parser.parse_args()
    if args.classifier:
        output = check_classifier()
    elif args.segmentation:
        output = check_segmentation()
    else:
        output = {**run_isolated("--classifier"), **run_isolated("--segmentation")}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
