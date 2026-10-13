#!/usr/bin/env python3
"""Evaluates deterministic intent classifier accuracy directly on text utterances.

Calculates:
- Action Sequence Exact Match Accuracy (order, intent type, and parameters match)
- Split Breakdown (train, dev, test)
- Category Breakdown
- Precision, Recall, and Confusion metrics for supported vs. rejected
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Any

from app.intents.classifier import IntentClassifier
from app.intents.models import CommandStatus


def actions_match(expected: List[Dict[str, Any]], actual_actions: List[Any]) -> bool:
    """Checks if predicted actions match ground truth in sequence, type, and parameters."""
    if len(expected) != len(actual_actions):
        return False

    for exp_act, act_obj in zip(expected, actual_actions):
        exp_type = exp_act.get("type")
        act_type = act_obj.type.value if hasattr(act_obj.type, "value") else str(act_obj.type)
        if exp_type != act_type:
            return False

        exp_params = exp_act.get("parameters", {})
        act_params = act_obj.parameters
        if exp_params != act_params:
            return False

    return True


def evaluate_text_dataset(
    filepath: str = "data/stt/commands.jsonl",
    target_split: str = "all",
) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        print(f"[ERROR] Dataset file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    classifier = IntentClassifier()

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if target_split != "all":
        records = [r for r in records if r.get("split") == target_split]

    total = len(records)
    if total == 0:
        print(f"[WARN] No records found for split '{target_split}'.")
        return {}

    correct = 0
    split_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    cat_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    failures = []

    # Confusion matrix: (expected_supported, actual_supported)
    # TP: supported=True and parsed valid
    # FP: supported=False but parsed valid
    # TN: supported=False and parsed rejected/unsupported
    # FN: supported=True but parsed rejected/unsupported
    tp = fp = tn = fn = 0

    for rec in records:
        rec_id = rec["id"]
        utt = rec["utterance"]
        exp_actions = rec["expected_actions"]
        cat = rec["category"]
        split = rec["split"]
        is_supported = rec["supported"]

        parsed = classifier.classify(utt)
        is_exact = actions_match(exp_actions, parsed.actions)

        actual_valid = parsed.status == CommandStatus.VALID and len(parsed.actions) > 0

        if is_supported:
            if actual_valid:
                tp += 1
            else:
                fn += 1
        else:
            if actual_valid:
                fp += 1
            else:
                tn += 1

        split_stats[split]["total"] += 1
        cat_stats[cat]["total"] += 1

        if is_exact:
            correct += 1
            split_stats[split]["correct"] += 1
            cat_stats[cat]["correct"] += 1
        else:
            failures.append({
                "id": rec_id,
                "utterance": utt,
                "split": split,
                "category": cat,
                "expected": exp_actions,
                "actual": [a.to_dict() for a in parsed.actions],
                "status": parsed.status.value,
                "error": parsed.error_message,
            })

    accuracy = (correct / total) * 100.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("=================================================================")
    print("           TEXT INTENT CLASSIFIER EVALUATION REPORT             ")
    print("=================================================================")
    print(f"Dataset File: {filepath}")
    print(f"Total Utterances Evaluated: {total}")
    print(f"Action Exact Match Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("\n------------------- ACCURACY BY SPLIT ---------------------------")
    for s, stats in sorted(split_stats.items()):
        s_acc = (stats["correct"] / stats["total"]) * 100.0
        print(f"  {s:8s}: {stats['correct']:3d}/{stats['total']:3d} ({s_acc:.2f}%)")

    print("\n------------------ ACCURACY BY CATEGORY -------------------------")
    for c, stats in sorted(cat_stats.items()):
        c_acc = (stats["correct"] / stats["total"]) * 100.0
        print(f"  {c:15s}: {stats['correct']:3d}/{stats['total']:3d} ({c_acc:.2f}%)")

    if failures:
        print(f"\n[FAILURES] {len(failures)} mismatch(es):")
        for f in failures[:15]:
            print(f"  [{f['id']} - {f['split']}] '{f['utterance']}'")
            print(f"     Expected: {f['expected']}")
            print(f"     Actual:   {f['actual']} (status={f['status']})")
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more failures.")
    else:
        print("\nAll evaluated utterances matched expected action sequences perfectly (100% Exact Match)!")

    print("=================================================================")

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "failures": failures,
    }


if __name__ == "__main__":
    split_arg = sys.argv[2] if len(sys.argv) > 2 else "all"
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/stt/commands.jsonl"
    evaluate_text_dataset(data_path, split_arg)
