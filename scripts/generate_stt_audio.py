#!/usr/bin/env python3
"""Batch audio generator for voice dataset using espeak-ng and ffmpeg.

Produces standard 16 kHz, 16-bit mono WAV audio clips matching the
exact requirements of whisper.cpp.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import List, Dict, Any


def generate_audio_for_record(
    record: Dict[str, Any],
    output_dir: str,
    temp_dir: str,
    overwrite: bool = False,
) -> bool:
    rec_id = record["id"]
    utterance = record["utterance"]
    out_wav = os.path.join(output_dir, f"{rec_id}.wav")

    if os.path.exists(out_wav) and not overwrite:
        return True

    temp_raw_wav = os.path.join(temp_dir, f"{rec_id}_raw.wav")

    try:
        # 1. Synthesize with espeak-ng
        res1 = subprocess.run(
            ["espeak-ng", "-w", temp_raw_wav, utterance],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if res1.returncode != 0 or not os.path.exists(temp_raw_wav):
            print(f"[ERROR] espeak-ng failed for {rec_id}: {res1.stderr.decode()}", file=sys.stderr)
            return False

        # 2. Transcode to 16 kHz mono 16-bit PCM WAV using ffmpeg
        res2 = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", temp_raw_wav,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                out_wav,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if res2.returncode != 0 or not os.path.exists(out_wav):
            print(f"[ERROR] ffmpeg failed for {rec_id}: {res2.stderr.decode()}", file=sys.stderr)
            return False

        return True
    finally:
        if os.path.exists(temp_raw_wav):
            try:
                os.remove(temp_raw_wav)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation audio using espeak-ng and ffmpeg.")
    parser.add_argument("--dataset", default="data/stt/commands.jsonl", help="Path to commands.jsonl")
    parser.add_argument("--output-dir", default="data/stt/audio", help="Output directory for WAV files")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"], help="Split to generate")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of audio files to generate")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing audio files")

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"[ERROR] Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.split != "all":
        records = [r for r in records if r.get("split") == args.split]

    if args.max_samples is not None:
        records = records[:args.max_samples]

    total = len(records)
    print(f"Generating synthetic audio for {total} utterances (split='{args.split}')...")

    success_count = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, rec in enumerate(records, start=1):
            ok = generate_audio_for_record(rec, args.output_dir, tmp_dir, args.overwrite)
            if ok:
                success_count += 1
            if i % 10 == 0 or i == total:
                print(f"  Progress: {i}/{total} files generated...")

    print(f"\n[DONE] Generated {success_count}/{total} audio files in {args.output_dir}")


if __name__ == "__main__":
    main()
