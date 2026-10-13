#!/usr/bin/env python3
"""Reproducible Offline SVM Intent Classifier Training & Evaluation Script.

Trains a Linear Support Vector Machine with TF-IDF features from scratch
using the in-vehicle voice assistant canonical dataset (data/stt/commands.jsonl).
Evaluates strictly on held-out validation (dev) and test splits without data leakage.

Evaluates two distinct layers separately (as required by specification):
  Layer A: Intent classifier (SVM predict accuracy, precision, recall, F1, confusion matrix)
  Layer B: End-to-end command handling (full pipeline matching requested actions, targets, conditions, negation, ordering)

Saves the trained model artifact and comprehensive evaluation report locally.
"""

import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# Ensure app is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.intents.models import IntentType, CommandStatus
from app.intents.classifier import IntentClassifier
from app.intents.normalizer import normalize_text, strip_polite_fillers
from app.intents.svm_pipeline import SVMIntentPipeline, build_svm_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainSVM")


def compute_file_hash(path: Path) -> str:
    """Computes SHA-256 hash of a file for reproducibility and provenance tracking."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dataset(dataset_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Loads JSONL dataset and calculates its SHA-256 digest."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    file_hash = compute_file_hash(dataset_path)
    return records, file_hash


def extract_atomic_samples(
    records: List[Dict[str, Any]],
    segmenter: IntentClassifier,
    is_train_split: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """Extracts text utterances and corresponding intent labels from dataset records.

    For evaluation splits (dev and test), evaluates single-action and unsupported
    utterances directly to prevent data leakage. For training split, also extracts
    atomic clauses segmented from train compound commands.

    Returns:
        (texts, labels, sample_ids)
    """
    texts: List[str] = []
    labels: List[str] = []
    sample_ids: List[str] = []

    for r in records:
        actions = r.get("expected_actions", [])
        rid = r.get("id", "")
        utterance = r.get("utterance", "").strip()

        if len(actions) == 1:
            intent_label = actions[0]["type"]
            texts.append(utterance)
            labels.append(intent_label)
            sample_ids.append(rid)

        elif len(actions) == 0:
            texts.append(utterance)
            labels.append("UNSUPPORTED")
            sample_ids.append(rid)

        elif is_train_split:
            # Segment compound utterances in train split into atomic clauses
            cleaned, _ = segmenter._preprocess_utterance(utterance)
            clauses = segmenter._segment_compound(cleaned)
            if len(clauses) == len(actions):
                for idx, (clause, act) in enumerate(zip(clauses, actions)):
                    texts.append(clause)
                    labels.append(act["type"])
                    sample_ids.append(f"{rid}_sub{idx+1}")
            else:
                texts.append(utterance)
                labels.append(actions[0]["type"])
                sample_ids.append(rid)

    return texts, labels, sample_ids


def evaluate_split(
    pipeline,
    texts: List[str],
    true_labels: List[str],
    all_classes: List[str],
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics on an atomic dataset split."""
    if not texts:
        return {}

    preds = pipeline.predict(texts)

    acc = float(accuracy_score(true_labels, preds))
    macro_p = float(precision_score(true_labels, preds, average="macro", zero_division=0))
    macro_r = float(recall_score(true_labels, preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(true_labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(true_labels, preds, average="weighted", zero_division=0))

    cm = confusion_matrix(true_labels, preds, labels=all_classes)

    cls_report = classification_report(
        true_labels,
        preds,
        labels=all_classes,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for cls_name in all_classes:
        metrics = cls_report.get(cls_name, {})
        per_class[cls_name] = {
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1_score": float(metrics.get("f1-score", 0.0)),
            "support": int(metrics.get("support", 0)),
        }

    return {
        "num_samples": len(texts),
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "predictions": list(preds),
        "true_labels": true_labels,
    }


def evaluate_layer_b_end_to_end(
    classifier: IntentClassifier,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluates Layer B: End-to-End Command Handling Pipeline.

    Verifies whether the complete pipeline:
      1. Correctly identifies single and compound requested actions in order
      2. Correctly extracts action entities and targets (e.g. door targets)
      3. Rejects unsupported or ambiguous requests safely
    """
    total = len(records)
    exact_matches = 0
    single_total = 0
    single_correct = 0
    compound_total = 0
    compound_correct = 0
    unsupported_total = 0
    unsupported_correct = 0

    mismatches = []

    for r in records:
        utterance = r.get("utterance", "")
        expected_actions = r.get("expected_actions", [])
        is_supported = r.get("supported", True)

        cmd = classifier.classify(utterance)

        # Check action match
        matched = False
        if not is_supported or len(expected_actions) == 0:
            unsupported_total += 1
            if cmd.status in (CommandStatus.UNSUPPORTED, CommandStatus.REJECTED, CommandStatus.AMBIGUOUS):
                unsupported_correct += 1
                matched = True
        else:
            if len(expected_actions) == 1:
                single_total += 1
            else:
                compound_total += 1

            if cmd.status == CommandStatus.VALID and len(cmd.actions) == len(expected_actions):
                all_match = True
                for act_pred, act_exp in zip(cmd.actions, expected_actions):
                    if act_pred.type.value != act_exp["type"]:
                        all_match = False
                        break
                    exp_params = act_exp.get("parameters", {})
                    for pk, pv in exp_params.items():
                        if act_pred.parameters.get(pk) != pv:
                            all_match = False
                            break
                if all_match:
                    matched = True
                    if len(expected_actions) == 1:
                        single_correct += 1
                    else:
                        compound_correct += 1

        if matched:
            exact_matches += 1
        else:
            mismatches.append({
                "id": r.get("id"),
                "utterance": utterance,
                "expected": expected_actions,
                "status": cmd.status.value,
                "predicted": [a.to_dict() for a in cmd.actions],
            })

    return {
        "total_records": total,
        "exact_matches": exact_matches,
        "accuracy": exact_matches / max(total, 1),
        "single_total": single_total,
        "single_accuracy": single_correct / max(single_total, 1),
        "compound_total": compound_total,
        "compound_accuracy": compound_correct / max(compound_total, 1),
        "unsupported_total": unsupported_total,
        "unsupported_accuracy": unsupported_correct / max(unsupported_total, 1),
        "num_mismatches": len(mismatches),
        "sample_mismatches": mismatches[:10],
    }


def measure_inference_latency(pipeline, sample_texts: List[str], repetitions: int = 100) -> Dict[str, float]:
    """Measures inference latency percentiles on the local hardware."""
    times_ms = []
    # Warmup
    for text in sample_texts[:10]:
        pipeline.predict([text])

    for _ in range(repetitions):
        for text in sample_texts:
            t0 = time.perf_counter()
            pipeline.predict([text])
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "mean_latency_ms": float(np.mean(arr)),
        "p50_latency_ms": float(np.percentile(arr, 50)),
        "p90_latency_ms": float(np.percentile(arr, 90)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "p99_latency_ms": float(np.percentile(arr, 99)),
        "min_latency_ms": float(np.min(arr)),
        "max_latency_ms": float(np.max(arr)),
        "total_measurements": len(times_ms),
    }


def main():
    dataset_path = REPO_ROOT / "data" / "stt" / "commands.jsonl"
    output_model_path = REPO_ROOT / "app" / "models" / "intent_svm.joblib"
    report_path = REPO_ROOT / "data" / "stt" / "svm_eval_report.json"

    logger.info("=" * 70)
    logger.info("TRAINING OFFLINE SVM INTENT CLASSIFIER (FROM SCRATCH)")
    logger.info("=" * 70)

    # 1. Load dataset
    records, data_hash = load_dataset(dataset_path)
    logger.info(f"Loaded {len(records)} records from {dataset_path.relative_to(REPO_ROOT)}")
    logger.info(f"Dataset SHA-256: {data_hash}")

    segmenter = IntentClassifier(enable_svm=False)

    train_records = [r for r in records if r.get("split") == "train"]
    dev_records = [r for r in records if r.get("split") == "dev"]
    test_records = [r for r in records if r.get("split") == "test"]

    logger.info(f"Splits: Train={len(train_records)}, Dev={len(dev_records)}, Test={len(test_records)}")

    # 2. Extract training samples
    X_train_raw, y_train, train_ids = extract_atomic_samples(train_records, segmenter, is_train_split=True)
    X_dev_raw, y_dev, dev_ids = extract_atomic_samples(dev_records, segmenter, is_train_split=False)
    X_test_raw, y_test, test_ids = extract_atomic_samples(test_records, segmenter, is_train_split=False)

    # Preprocessing with normalizer
    X_train = [strip_polite_fillers(normalize_text(x)) for x in X_train_raw]
    X_dev = [strip_polite_fillers(normalize_text(x)) for x in X_dev_raw]
    X_test = [strip_polite_fillers(normalize_text(x)) for x in X_test_raw]

    class_counts = Counter(y_train)
    all_classes = sorted(list(class_counts.keys()))

    logger.info(f"Extracted {len(X_train)} training samples across {len(all_classes)} classes:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cls_name:20}: {count:3} samples")

    logger.info(f"Extracted {len(X_dev)} dev samples, {len(X_test)} test samples.")

    # 3. Build & Train Pipeline with FeatureUnion (word unigrams/bigrams + char_wb 3-5 ngrams)
    c_param = 1.5
    ngram_range = (1, 2)
    char_ngram_range = (3, 5)
    random_state = 42

    logger.info(f"Constructing Pipeline: Word(1,2) + Char_wb(3,5) + LinearSVC(C={c_param})")
    pipeline = build_svm_pipeline(
        ngram_range=ngram_range,
        char_ngram_range=char_ngram_range,
        c_param=c_param,
        random_state=random_state,
    )

    t_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    t_train = time.perf_counter() - t_start
    logger.info(f"Model training completed in {t_train:.4f} seconds.")

    # 4. Evaluate Layer A: Intent Classifier
    logger.info("Evaluating Layer A: SVM Intent Classifier...")
    train_metrics = evaluate_split(pipeline, X_train, y_train, all_classes)
    dev_metrics = evaluate_split(pipeline, X_dev, y_dev, all_classes)
    test_metrics = evaluate_split(pipeline, X_test, y_test, all_classes)

    logger.info(f"Layer A Train Accuracy: {train_metrics['accuracy']:.4f} | F1 (Macro): {train_metrics['macro_f1']:.4f}")
    logger.info(f"Layer A Dev   Accuracy: {dev_metrics['accuracy']:.4f} | F1 (Macro): {dev_metrics['macro_f1']:.4f}")
    logger.info(f"Layer A Test  Accuracy: {test_metrics['accuracy']:.4f} | F1 (Macro): {test_metrics['macro_f1']:.4f}")

    # 5. Measure latency
    logger.info("Measuring inference latency on test samples...")
    latency_stats = measure_inference_latency(pipeline, X_test[:20], repetitions=50)
    logger.info(f"Latency: Mean={latency_stats['mean_latency_ms']:.2f}ms | p50={latency_stats['p50_latency_ms']:.2f}ms | p95={latency_stats['p95_latency_ms']:.2f}ms")

    # 6. Save model artifact
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    svm_wrapper = SVMIntentPipeline(output_model_path)
    svm_wrapper.pipeline = pipeline
    svm_wrapper.classes = all_classes

    metadata = {
        "dataset_file": str(dataset_path.name),
        "dataset_hash": data_hash,
        "total_dataset_records": len(records),
        "train_samples": len(X_train),
        "dev_samples": len(X_dev),
        "test_samples": len(X_test),
        "classes": all_classes,
        "class_counts_train": dict(class_counts),
        "hyperparameters": {
            "ngram_range": list(ngram_range),
            "char_ngram_range": list(char_ngram_range),
            "c_param": c_param,
            "class_weight": "balanced",
            "random_state": random_state,
        },
        "training_duration_seconds": round(t_train, 4),
        "latency_ms": latency_stats,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
        "dev_metrics": {
            "accuracy": dev_metrics["accuracy"],
            "macro_precision": dev_metrics["macro_precision"],
            "macro_recall": dev_metrics["macro_recall"],
            "macro_f1": dev_metrics["macro_f1"],
            "weighted_f1": dev_metrics["weighted_f1"],
        },
    }

    svm_wrapper.save(output_model_path, metadata)
    model_size_bytes = output_model_path.stat().st_size
    logger.info(f"Saved model artifact: {output_model_path.relative_to(REPO_ROOT)} ({model_size_bytes / 1024:.2f} KB)")

    # 7. Evaluate Layer B: End-to-End Command Handling Pipeline
    logger.info("Evaluating Layer B: End-to-End Command Handling Pipeline...")
    full_classifier = IntentClassifier(model_path=output_model_path, enable_svm=True)
    layer_b_test = evaluate_layer_b_end_to_end(full_classifier, test_records)
    layer_b_dev = evaluate_layer_b_end_to_end(full_classifier, dev_records)
    logger.info(f"Layer B Test Overall Accuracy: {layer_b_test['accuracy'] * 100:.2f}% (Single: {layer_b_test['single_accuracy']*100:.1f}%, Compound: {layer_b_test['compound_accuracy']*100:.1f}%, Rejection: {layer_b_test['unsupported_accuracy']*100:.1f}%)")

    # 8. Write complete held-out evaluation report
    full_report = {
        "metadata": metadata,
        "layer_a_svm_test": test_metrics,
        "layer_a_svm_dev": dev_metrics,
        "layer_a_svm_train": {
            "num_samples": train_metrics["num_samples"],
            "accuracy": train_metrics["accuracy"],
            "macro_f1": train_metrics["macro_f1"],
            "weighted_f1": train_metrics["weighted_f1"],
        },
        "layer_b_end_to_end_test": layer_b_test,
        "layer_b_end_to_end_dev": layer_b_dev,
        "classes": all_classes,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Saved comprehensive evaluation report: {report_path.relative_to(REPO_ROOT)}")

    # 9. Print clean summary
    print("\n" + "=" * 70)
    print("OFFLINE SVM INTENT CLASSIFIER & COMMAND PIPELINE REPORT")
    print("=" * 70)
    print(f"Dataset Records:        {len(records)} (Train: {len(train_records)}, Dev: {len(dev_records)}, Test: {len(test_records)})")
    print(f"Learned Intent Classes: {len(all_classes)}")
    print(f"Training Duration:      {t_train:.4f}s")
    print(f"Model Artifact Size:    {model_size_bytes / 1024:.2f} KB")
    print(f"Inference Latency:      Mean: {latency_stats['mean_latency_ms']:.2f} ms | p95: {latency_stats['p95_latency_ms']:.2f} ms")
    print("-" * 70)
    print(f"LAYER A (SVM INTENT PREDICTION) - HELD-OUT TEST METRICS:")
    print(f"  Accuracy:             {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro Precision:      {test_metrics['macro_precision'] * 100:.2f}%")
    print(f"  Macro Recall:         {test_metrics['macro_recall'] * 100:.2f}%")
    print(f"  Macro F1-Score:       {test_metrics['macro_f1'] * 100:.2f}%")
    print(f"  Weighted F1-Score:    {test_metrics['weighted_f1'] * 100:.2f}%")
    print("-" * 70)
    print(f"LAYER B (END-TO-END PIPELINE) - HELD-OUT TEST METRICS:")
    print(f"  Overall Command Acc:  {layer_b_test['accuracy'] * 100:.2f}%")
    print(f"  Single Action Acc:    {layer_b_test['single_accuracy'] * 100:.2f}% ({layer_b_test['single_total']} samples)")
    print(f"  Compound Action Acc:  {layer_b_test['compound_accuracy'] * 100:.2f}% ({layer_b_test['compound_total']} samples)")
    print(f"  Unsupported Reject:   {layer_b_test['unsupported_accuracy'] * 100:.2f}% ({layer_b_test['unsupported_total']} samples)")
    print("=" * 70)


if __name__ == "__main__":
    main()
