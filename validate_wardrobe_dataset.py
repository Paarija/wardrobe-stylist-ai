"""Validate reviewed wardrobe-photo manifests and prevent split leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    "image", "label", "garment_id", "capture_session", "split",
    "reviewed", "source", "consent",
}
LABELS = {"top", "bottom", "shoes", "dress"}
SPLITS = {"train", "development", "test"}


def validate_manifest(path: Path) -> dict:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            if row["label"].strip().lower() not in LABELS:
                raise ValueError(f"Line {line}: invalid label {row['label']!r}.")
            if row["split"].strip().lower() not in SPLITS:
                raise ValueError(f"Line {line}: split must be train, development, or test.")
            if row["reviewed"].strip().lower() != "true":
                raise ValueError(f"Line {line}: every included label must be human reviewed.")
            if row["consent"].strip().lower() != "true":
                raise ValueError(f"Line {line}: documented consent is required.")
            image_path = (path.parent / row["image"].strip()).resolve()
            if not image_path.is_file():
                raise ValueError(f"Line {line}: missing image {image_path}.")
            rows.append({**row, "line": line, "path": image_path,
                         "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
    if not rows:
        raise ValueError("The manifest has no photo rows.")

    for group_name in ("garment_id", "capture_session", "sha256"):
        memberships = defaultdict(set)
        for row in rows:
            memberships[row[group_name]].add(row["split"].strip().lower())
        leaked = [value for value, splits in memberships.items() if len(splits) > 1]
        if leaked:
            raise ValueError(f"{group_name} crosses dataset splits: {leaked[:5]}")

    split_counts = Counter(row["split"].strip().lower() for row in rows)
    if set(split_counts) != SPLITS:
        raise ValueError("The manifest must contain train, development, and final test photos.")
    return {
        "manifest": str(path),
        "photos": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(Counter(row["label"].strip().lower() for row in rows).items())),
        "sources": dict(sorted(Counter(row["source"].strip() for row in rows).items())),
        "unique_garments": len({row["garment_id"] for row in rows}),
        "unique_capture_sessions": len({row["capture_session"] for row in rows}),
        "duplicate_sha256_images": len(rows) - len({row["sha256"] for row in rows}),
        "test_split_locked_for_final_evaluation": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a wardrobe-photo dataset manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_manifest(args.manifest)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
