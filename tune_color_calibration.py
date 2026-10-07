"""Tune personal LAB correction radius with leave-one-item-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

import fashion_ai.color as color_model
from fashion_ai.db import init_db, list_color_feedback, list_items
from fashion_ai.recommender import COLORS


def evaluate(user_id: int, thresholds: tuple[float, ...]) -> dict:
    items = [
        item for item in list_items(user_id)
        if item["name"].split()[0].lower() in COLORS
    ]
    feedback = list_color_feedback(user_id)
    if len(items) < 4 or len(feedback) < 4:
        raise ValueError("At least four named items with colour corrections are required.")
    results = []
    for threshold in thresholds:
        color_model.CALIBRATION_MAX_DISTANCE = threshold
        correct = 0
        for item in items:
            held_out = [entry for entry in feedback if entry["item_id"] != item["id"]]
            with Image.open(item["image_path"]) as source:
                predicted = color_model.infer_color(source.convert("RGB"), held_out)
            correct += predicted == item["name"].split()[0].lower()
        results.append({
            "threshold": threshold,
            "correct": correct,
            "images": len(items),
            "leave_one_item_out_accuracy": round(correct / len(items), 4),
        })
    best_accuracy = max(result["leave_one_item_out_accuracy"] for result in results)
    selected = max(result["threshold"] for result in results
                   if result["leave_one_item_out_accuracy"] == best_accuracy)
    return {
        "protocol": "leave one wardrobe item out of personal colour calibration",
        "labels": "first colour word in the user-reviewed item name",
        "selected_threshold": selected,
        "results": results,
        "limitations": "Small personal calibration set; this is not a public benchmark.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    init_db()
    report = evaluate(args.user_id, (1, 2, 3, 4, 5, 6, 8, 10, 12))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
