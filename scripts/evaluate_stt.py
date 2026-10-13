#!/usr/bin/env python3
"""End-to-end Speech-to-Text (STT) and Intent benchmark runner using whisper.cpp.

Calculates:
1. Word Error Rate (WER) using Levenshtein distance on tokenized words.
2. Transcript Exact Match Accuracy (normalized hypothesis == normalized reference).
3. Action Sequence Exact Match Accuracy (predicted actions == expected actions).
4. Average transcription latency (seconds).
5. Detailed JSON report saved to data/stt/eval_report.json.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Any, Tuple

from app.intents.classifier import IntentClassifier
from app.intents.normalizer import normalize_text
from app.speech.transcriber import WhisperTranscriber
from scripts.eval_intents_text import actions_match
from scripts.generate_stt_audio import generate_audio_for_record


def compute_wer(reference: str, hypothesis: str) -> float:
    """Calculates Levenshtein-based Word Error Rate (WER) between reference and hypothesis."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    r_len = len(ref_words)
    h_len = len(hyp_words)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    return dp[r_len][h_len] / r_len


def run_stt_evaluation(
    dataset_path: str = "data/stt/commands.jsonl",
    audio_dir: str = "data/stt/audio",
    split: str = "test",
    max_samples: int = None,
    report_file: str = "data/stt/eval_report.json",
) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(audio_dir, exist_ok=True)

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if split != "all":
        records = [r for r in records if r.get("split") == split]

    if max_samples is not None:
        records = records[:max_samples]

    total = len(records)
    if total == 0:
        print(f"[ERROR] No records found for split '{split}'", file=sys.stderr)
        sys.exit(1)

    print(f"\n=================================================================")
    print(f"       WHISPER.CPP SPEECH RECOGNITION & INTENT BENCHMARK        ")
    print(f"=================================================================")
    print(f"Dataset: {dataset_path} (Split: {split})")
    print(f"Total Samples to Evaluate: {total}")
    print(f"Model: whisper.cpp / ggml-tiny.en.bin")
    print(f"Audio Format: 16 kHz, 16-bit Mono WAV\n")

    transcriber = WhisperTranscriber()
    classifier = IntentClassifier()

    sample_results = []
    total_wer = 0.0
    transcript_exact_matches = 0
    action_exact_matches = 0
    total_latency = 0.0

    category_stats = defaultdict(lambda: {
        "total": 0,
        "action_match": 0,
        "transcript_match": 0,
        "total_wer": 0.0,
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, rec in enumerate(records, start=1):
            rec_id = rec["id"]
            ref_utt = rec["utterance"]
            ref_norm = rec["normalized_utterance"]
            exp_actions = rec["expected_actions"]
            cat = rec["category"]

            wav_path = os.path.join(audio_dir, f"{rec_id}.wav")
            if not os.path.exists(wav_path):
                ok = generate_audio_for_record(rec, audio_dir, tmp_dir)
                if not ok:
                    print(f"[WARN] Failed to generate audio for {rec_id}, skipping.")
                    continue

            # 1. Transcribe audio with whisper.cpp
            t_res = transcriber.transcribe(wav_path)
            hyp_raw = t_res.text
            hyp_norm = normalize_text(hyp_raw)
            latency = t_res.latency_sec
            total_latency += latency

            # 2. Compute WER
            wer = compute_wer(ref_norm, hyp_norm)
            total_wer += wer

            # 3. Compute Transcript Match
            is_transcript_match = (ref_norm == hyp_norm)
            if is_transcript_match:
                transcript_exact_matches += 1

            # 4. Classify transcribed utterance and compute Action Match
            parsed = classifier.classify(hyp_raw)
            is_action_match = actions_match(exp_actions, parsed.actions)
            if is_action_match:
                action_exact_matches += 1

            # Update category stats
            category_stats[cat]["total"] += 1
            if is_action_match:
                category_stats[cat]["action_match"] += 1
            if is_transcript_match:
                category_stats[cat]["transcript_match"] += 1
            category_stats[cat]["total_wer"] += wer

            sample_entry = {
                "id": rec_id,
                "category": cat,
                "reference": ref_utt,
                "reference_normalized": ref_norm,
                "transcription_raw": hyp_raw,
                "transcription_normalized": hyp_norm,
                "wer": round(wer, 4),
                "transcript_match": is_transcript_match,
                "action_match": is_action_match,
                "expected_actions": exp_actions,
                "predicted_actions": [a.to_dict() for a in parsed.actions],
                "status": parsed.status.value,
                "latency_sec": latency,
            }
            sample_results.append(sample_entry)

            status_symbol = "✓" if is_action_match else "✗"
            print(f"[{idx:2d}/{total:2d}] {status_symbol} {rec_id} ({cat:11s}) | WER: {wer:.2f} | Latency: {latency:.2f}s | Hyp: '{hyp_norm}'")

    avg_wer = (total_wer / total) * 100.0
    transcript_acc = (transcript_exact_matches / total) * 100.0
    action_acc = (action_exact_matches / total) * 100.0
    avg_latency = total_latency / total

    print("\n=================================================================")
    print("                    FINAL BENCHMARK RESULTS                      ")
    print("=================================================================")
    print(f"Total Evaluated: {total}")
    print(f"Word Error Rate (WER):             {avg_wer:.2f}%")
    print(f"Transcript Exact Match:            {transcript_exact_matches}/{total} ({transcript_acc:.2f}%)")
    print(f"Action Sequence Exact Match:       {action_exact_matches}/{total} ({action_acc:.2f}%)")
    print(f"Mean STT Latency:                  {avg_latency:.3f} seconds / sample")
    print("\n-------------------- BREAKDOWN BY CATEGORY ----------------------")
    for cat, stats in sorted(category_stats.items()):
        c_tot = stats["total"]
        c_act_acc = (stats["action_match"] / c_tot) * 100.0
        c_trans_acc = (stats["transcript_match"] / c_tot) * 100.0
        c_wer = (stats["total_wer"] / c_tot) * 100.0
        print(f"  {cat:15s}: Action Match: {stats['action_match']:2d}/{c_tot:2d} ({c_act_acc:6.2f}%) | Transcript: {c_trans_acc:6.2f}% | WER: {c_wer:5.2f}%")

    report_data = {
        "benchmark": "In-Vehicle Voice Assistant STT & Compound Action Benchmark",
        "split": split,
        "total_samples": total,
        "overall_metrics": {
            "mean_wer_percent": round(avg_wer, 2),
            "transcript_exact_match_percent": round(transcript_acc, 2),
            "action_exact_match_percent": round(action_acc, 2),
            "mean_latency_sec": round(avg_latency, 3),
        },
        "category_metrics": {
            cat: {
                "total": stats["total"],
                "action_match": stats["action_match"],
                "action_accuracy_percent": round((stats["action_match"] / stats["total"]) * 100.0, 2),
                "transcript_match": stats["transcript_match"],
                "transcript_accuracy_percent": round((stats["transcript_match"] / stats["total"]) * 100.0, 2),
                "mean_wer_percent": round((stats["total_wer"] / stats["total"]) * 100.0, 2),
            }
            for cat, stats in category_stats.items()
        },
        "samples": sample_results,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed evaluation report saved to: {report_file}")
    print("=================================================================\n")

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate whisper.cpp speech recognition on voice dataset.")
    parser.add_argument("--dataset", default="data/stt/commands.jsonl", help="Path to commands.jsonl")
    parser.add_argument("--audio-dir", default="data/stt/audio", help="Directory of audio WAV files")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"], help="Split to evaluate")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--report-file", default="data/stt/eval_report.json", help="Output JSON report file")

    args = parser.parse_args()
    run_stt_evaluation(
        dataset_path=args.dataset,
        audio_dir=args.audio_dir,
        split=args.split,
        max_samples=args.max_samples,
        report_file=args.report_file,
    )


if __name__ == "__main__":
    main()
