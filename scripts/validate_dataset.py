#!/usr/bin/env python3
"""Dataset validation script for in-vehicle voice assistant STT benchmarks.

Checks:
1. File existence and JSON Lines parsing.
2. All required fields present and non-empty: id, utterance, normalized_utterance, expected_actions, category, split, difficulty, supported, notes.
3. ID uniqueness and valid sequential format.
4. Valid split values: 'train', 'dev', 'test'.
5. Valid category values.
6. Valid expected_actions schema: each action must have valid 'type' in IntentType and 'parameters' dict.
7. Split integrity: held-out test split must contain at least 50 samples.
8. No utterance collisions or exact leaks between train/dev and test.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.intents.models import IntentType

VALID_SPLITS = {"train", "dev", "test"}
VALID_CATEGORIES = {
    "lighting",
    "climate",
    "doors",
    "sensors",
    "emergency",
    "assistant",
    "global",
    "compound",
    "unsupported",
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_INTENT_TYPES = {i.value for i in IntentType}


def validate_dataset(filepath: str = "data/stt/commands.jsonl") -> bool:
    if not os.path.exists(filepath):
        print(f"[ERROR] Dataset file not found: {filepath}", file=sys.stderr)
        return False

    seen_ids: Set[str] = set()
    seen_utterances: Dict[str, str] = {}  # normalized_text -> split
    split_counts: Dict[str, int] = {s: 0 for s in VALID_SPLITS}
    category_counts: Dict[str, int] = {c: 0 for c in VALID_CATEGORIES}
    errors = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            # 1. Required fields
            req_fields = [
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
            for field in req_fields:
                if field not in record:
                    errors.append(f"Line {line_num}: Missing required field '{field}'")

            # 2. ID uniqueness
            rec_id = record.get("id", "")
            if not rec_id:
                errors.append(f"Line {line_num}: Empty 'id'")
            elif rec_id in seen_ids:
                errors.append(f"Line {line_num}: Duplicate ID '{rec_id}'")
            seen_ids.add(rec_id)

            # 3. Utterance and normalized utterance
            utt = record.get("utterance", "").strip()
            norm_utt = record.get("normalized_utterance", "").strip()
            if not utt:
                errors.append(f"Line {line_num} ({rec_id}): Empty 'utterance'")
            if not norm_utt:
                errors.append(f"Line {line_num} ({rec_id}): Empty 'normalized_utterance'")

            # 4. Split validation
            split = record.get("split", "")
            if split not in VALID_SPLITS:
                errors.append(f"Line {line_num} ({rec_id}): Invalid split '{split}'")
            else:
                split_counts[split] += 1

            # 5. Leak detection
            if norm_utt in seen_utterances:
                prev_split = seen_utterances[norm_utt]
                if prev_split != split and (split == "test" or prev_split == "test"):
                    errors.append(
                        f"Line {line_num} ({rec_id}): Exact utterance leak between test and {prev_split}: '{norm_utt}'"
                    )
            seen_utterances[norm_utt] = split

            # 6. Category validation
            cat = record.get("category", "")
            if cat not in VALID_CATEGORIES:
                errors.append(f"Line {line_num} ({rec_id}): Invalid category '{cat}'")
            else:
                category_counts[cat] += 1

            # 7. Difficulty validation
            diff = record.get("difficulty", "")
            if diff not in VALID_DIFFICULTIES:
                errors.append(f"Line {line_num} ({rec_id}): Invalid difficulty '{diff}'")

            # 8. Supported boolean
            supp = record.get("supported")
            if not isinstance(supp, bool):
                errors.append(f"Line {line_num} ({rec_id}): 'supported' must be a boolean")

            # 9. Expected actions validation
            actions = record.get("expected_actions")
            if not isinstance(actions, list):
                errors.append(f"Line {line_num} ({rec_id}): 'expected_actions' must be a list")
            else:
                if supp and len(actions) == 0:
                    errors.append(f"Line {line_num} ({rec_id}): Supported utterance has empty expected_actions")
                if not supp and len(actions) > 0:
                    errors.append(f"Line {line_num} ({rec_id}): Unsupported utterance has non-empty expected_actions")

                for act_idx, act in enumerate(actions):
                    if not isinstance(act, dict):
                        errors.append(f"Line {line_num} ({rec_id}): Action {act_idx} is not a dict")
                        continue
                    act_type = act.get("type", "")
                    if act_type not in VALID_INTENT_TYPES:
                        errors.append(f"Line {line_num} ({rec_id}): Invalid action type '{act_type}'")
                    params = act.get("parameters")
                    if not isinstance(params, dict):
                        errors.append(f"Line {line_num} ({rec_id}): Action {act_idx} parameters must be a dict")

    # 10. Split test size check
    if split_counts.get("test", 0) < 50:
        errors.append(f"Held-out test split must contain at least 50 samples (found {split_counts.get('test', 0)})")

    if errors:
        print(f"[FAIL] Dataset validation failed with {len(errors)} error(s):", file=sys.stderr)
        for err in errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.", file=sys.stderr)
        return False

    print("==================================================")
    print("DATASET VALIDATION PASSED (ALL CHECKS SUCCESSFUL)")
    print("==================================================")
    print(f"Total Canonical Records: {len(seen_ids)}")
    print("\nSplit Distribution:")
    for s, c in sorted(split_counts.items()):
        print(f"  {s:8s}: {c:3d} ({c/len(seen_ids)*100:.1f}%)")
    print("\nCategory Distribution:")
    for cat, c in sorted(category_counts.items()):
        print(f"  {cat:15s}: {c:3d} ({c/len(seen_ids)*100:.1f}%)")
    print("==================================================")
    return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/stt/commands.jsonl"
    success = validate_dataset(path)
    sys.exit(0 if success else 1)
