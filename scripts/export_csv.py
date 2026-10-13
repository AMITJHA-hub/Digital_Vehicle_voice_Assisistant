#!/usr/bin/env python3
"""Export utility to convert data/stt/commands.jsonl to data/stt/commands.csv."""

import csv
import json
import os
import sys


def export_jsonl_to_csv(
    jsonl_path: str = "data/stt/commands.jsonl",
    csv_path: str = "data/stt/commands.csv",
) -> None:
    if not os.path.exists(jsonl_path):
        print(f"[ERROR] Source JSONL not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "id",
        "utterance",
        "normalized_utterance",
        "expected_actions",
        "category",
        "split",
        "difficulty",
        "supported",
        "notes",
    ]

    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as in_f, \
         open(csv_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row = dict(rec)
            row["expected_actions"] = json.dumps(rec.get("expected_actions", []))
            writer.writerow(row)
            count += 1

    print(f"Successfully exported {count} records from {jsonl_path} to {csv_path}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/stt/commands.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/stt/commands.csv"
    export_jsonl_to_csv(src, dst)
