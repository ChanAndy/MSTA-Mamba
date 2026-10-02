#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate schema and patient-level split isolation")
    parser.add_argument("manifest")
    args = parser.parse_args()
    path = Path(args.manifest)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video_id", "video_path", "split", "patient_id", "labels"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    patients: dict[str, set[str]] = defaultdict(set)
    videos = set()
    for number, row in enumerate(rows, start=2):
        if row["video_id"] in videos:
            raise SystemExit(f"Duplicate video_id at row {number}: {row['video_id']}")
        videos.add(row["video_id"])
        split = row["split"].lower()
        if split not in {"train", "val", "test"}:
            raise SystemExit(f"Invalid split at row {number}: {split}")
        patients[row["patient_id"]].add(split)
        labels = json.loads(row["labels"])
        if not labels or any(not isinstance(value, int) or value < 0 for value in labels):
            raise SystemExit(f"Invalid labels at row {number}")
    leakage = {patient: splits for patient, splits in patients.items() if len(splits) > 1}
    if leakage:
        preview = list(leakage.items())[:10]
        raise SystemExit(f"Patient-level split leakage ({len(leakage)} patients): {preview}")
    counts = {split: sum(row["split"].lower() == split for row in rows) for split in ("train", "val", "test")}
    print(f"OK: {len(rows)} videos, {len(patients)} patients, splits={counts}")


if __name__ == "__main__":
    main()

