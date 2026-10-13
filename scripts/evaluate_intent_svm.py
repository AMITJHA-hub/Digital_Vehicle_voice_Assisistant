#!/usr/bin/env python3
"""Offline SVM Prediction & Evaluation CLI Tool.

Supports:
  1. Evaluating saved SVM model on held-out test or dev splits
  2. Single utterance intent prediction with confidence and decision margins
  3. Interactive REPL for real-time offline testing
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.intents.classifier import IntentClassifier
from app.intents.svm_pipeline import SVMIntentPipeline


def evaluate_split(split_name: str):
    report_file = REPO_ROOT / "data" / "stt" / "svm_eval_report.json"
    if not report_file.exists():
        print(f"Error: Evaluation report not found at {report_file}. Run scripts/train_intent_svm.py first.")
        sys.exit(1)

    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    meta = rep.get("metadata", {})
    layer_a_key = f"layer_a_svm_{split_name}"
    layer_b_key = f"layer_b_end_to_end_{split_name}"

    la = rep.get(layer_a_key, {})
    lb = rep.get(layer_b_key, {})

    print("=" * 65)
    print(f"EVALUATION RESULTS: {split_name.upper()} SPLIT")
    print("=" * 65)
    print(f"Model File:          {meta.get('model_file', 'intent_svm.joblib')}")
    print(f"Dataset Hash:        {meta.get('dataset_hash', 'N/A')[:16]}...")
    print(f"Learned Classes:     {len(meta.get('classes', []))}")
    print("-" * 65)
    if la:
        print("LAYER A: ATOMIC SVM INTENT PREDICTION")
        print(f"  Samples:           {la.get('num_samples')}")
        print(f"  Accuracy:          {la.get('accuracy', 0.0) * 100:.2f}%")
        print(f"  Macro Precision:   {la.get('macro_precision', 0.0) * 100:.2f}%")
        print(f"  Macro Recall:      {la.get('macro_recall', 0.0) * 100:.2f}%")
        print(f"  Macro F1:          {la.get('macro_f1', 0.0) * 100:.2f}%")
        print(f"  Weighted F1:       {la.get('weighted_f1', 0.0) * 100:.2f}%")
    print("-" * 65)
    if lb:
        print("LAYER B: END-TO-END COMMAND PIPELINE")
        print(f"  Total Records:     {lb.get('total_records')}")
        print(f"  Overall Match:     {lb.get('accuracy', 0.0) * 100:.2f}%")
        print(f"  Single Action:     {lb.get('single_accuracy', 0.0) * 100:.2f}%")
        print(f"  Compound Action:   {lb.get('compound_accuracy', 0.0) * 100:.2f}%")
        print(f"  Unsupported Reject:{lb.get('unsupported_accuracy', 0.0) * 100:.2f}%")
    print("=" * 65)


def predict_single(text: str):
    classifier = IntentClassifier(enable_svm=True)
    svm = classifier.svm_pipeline

    print("-" * 65)
    print(f"Input Utterance:   '{text}'")
    print(f"SVM Active:        {classifier.is_svm_active}")

    if svm and svm.is_loaded:
        pred = svm.predict(text)
        print(f"SVM Intent:        {pred.intent.value}")
        print(f"SVM Confidence:    {pred.confidence:.4f}")
        print(f"Top Margin:        {pred.top_margin:.3f}")
        print(f"Margin Gap:        {pred.margin_gap:.3f}")
        print(f"Confident:         {pred.is_confident}")

    parsed = classifier.classify(text)
    print(f"Pipeline Status:   {parsed.status.value}")
    print(f"Pipeline Intent:   {parsed.intent.value}")
    print(f"Actions ({len(parsed.actions)}):")
    for idx, act in enumerate(parsed.actions, start=1):
        params_str = json.dumps(act.parameters)
        print(f"  [{idx}] {act.type.value} -> {params_str}")
    if parsed.error_message:
        print(f"Message:           {parsed.error_message}")
    print("-" * 65)


def interactive_mode():
    classifier = IntentClassifier(enable_svm=True)
    print("In-Vehicle Offline SVM Intent REPL (type 'exit' or Ctrl+C to quit)")
    print(f"SVM Status: {'ONLINE (Active)' if classifier.is_svm_active else 'FALLBACK (Rules)'}")
    print("-" * 65)

    while True:
        try:
            line = input("Voice Transcript > ").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                break
            predict_single(line)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Evaluate or test offline SVM intent classifier.")
    parser.add_argument("--predict", type=str, help="Single utterance text to classify.")
    parser.add_argument("--split", choices=["test", "dev", "train"], default=None, help="Display evaluation report for split.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI REPL.")

    args = parser.parse_args()

    if args.predict:
        predict_single(args.predict)
    elif args.split:
        evaluate_split(args.split)
    elif args.interactive:
        interactive_mode()
    else:
        evaluate_split("test")


if __name__ == "__main__":
    main()
