import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from app.config import config, AppConfig
from app.audio import TextToSpeech
from app.hardware import get_hardware
from app.intents import IntentClassifier
from app.assistant import CommandDispatcher
from app.speech import WhisperTranscriber


def run_pipeline(
    audio_path: Path,
    transcriber: WhisperTranscriber,
    classifier: IntentClassifier,
    dispatcher: CommandDispatcher,
    tts: Optional[TextToSpeech] = None,
) -> dict:
    """Executes the full pipeline: Audio WAV -> whisper.cpp -> Intent -> Action -> Response."""
    start_total = time.perf_counter()

    # 1. Transcribe audio
    trans_res = transcriber.transcribe(audio_path)
    if not trans_res.success:
        return {
            "file": audio_path.name,
            "success": False,
            "error": trans_res.error,
            "transcription": "",
            "intent": "ERROR",
            "response": f"Speech recognition failed: {trans_res.error}",
            "stt_latency_sec": trans_res.latency_sec,
            "total_latency_sec": round(time.perf_counter() - start_total, 3),
        }

    # 2. Classify intent
    start_nlp = time.perf_counter()
    cmd = classifier.classify(trans_res.text)
    nlp_latency = time.perf_counter() - start_nlp

    # 3. Dispatch to vehicle hardware
    start_dispatch = time.perf_counter()
    disp_res = dispatcher.dispatch(cmd)
    dispatch_latency = time.perf_counter() - start_dispatch

    # 4. Optional spoken text-to-speech feedback
    if tts and tts.enabled:
        tts.speak(disp_res.response_text, play_audio=True)

    total_latency = time.perf_counter() - start_total

    return {
        "file": audio_path.name,
        "success": disp_res.success,
        "error": None,
        "transcription": trans_res.text,
        "intent": cmd.intent.value,
        "entities": cmd.entities,
        "response": disp_res.response_text,
        "should_exit": disp_res.should_exit,
        "should_stop_listening": disp_res.should_stop_listening,
        "stt_latency_sec": trans_res.latency_sec,
        "nlp_latency_ms": round(nlp_latency * 1000, 2),
        "dispatch_latency_ms": round(dispatch_latency * 1000, 2),
        "total_latency_sec": round(total_latency, 3),
    }


def print_result_card(res: dict) -> None:
    """Prints a clear, formatted console report for an executed command."""
    print("=" * 60)
    print(f" Audio Input : {res.get('file', 'N/A')}")
    print(f" Transcript  : \"{res.get('transcription', '')}\"")
    print(f" Intent      : {res.get('intent', 'UNKNOWN')}")
    if res.get("entities"):
        print(f" Entities    : {res.get('entities')}")
    print(f" Response    : {res.get('response', '')}")
    print(f" Latency     : STT {res.get('stt_latency_sec', 0):.3f}s | "
          f"NLP {res.get('nlp_latency_ms', 0):.1f}ms | "
          f"Total {res.get('total_latency_sec', 0):.3f}s")
    print("=" * 60)


def interactive_loop(
    transcriber: WhisperTranscriber,
    classifier: IntentClassifier,
    dispatcher: CommandDispatcher,
    tts: Optional[TextToSpeech] = None,
) -> None:
    """Push-to-enter interactive command shell."""
    print("\n--- In-Vehicle Voice Assistant Interactive Shell ---")
    print("Enter a text command directly, or provide a path to a .wav audio file.")
    print("Type 'exit' or 'quit' to terminate.\n")

    while True:
        try:
            user_input = input("Vehicle-Voice> ").strip()
            if not user_input:
                continue

            input_path = Path(user_input)
            if input_path.is_file() and input_path.suffix.lower() == ".wav":
                res = run_pipeline(input_path, transcriber, classifier, dispatcher, tts)
                print_result_card(res)
                if res.get("should_exit"):
                    break
            else:
                # Direct text command processing
                start = time.perf_counter()
                cmd = classifier.classify(user_input)
                disp_res = dispatcher.dispatch(cmd)
                latency = time.perf_counter() - start

                if tts and tts.enabled:
                    tts.speak(disp_res.response_text, play_audio=True)

                print("-" * 50)
                print(f"Input    : \"{user_input}\"")
                print(f"Intent   : {cmd.intent.value}")
                print(f"Response : {disp_res.response_text}")
                print(f"Latency  : {latency * 1000:.2f} ms")
                print("-" * 50)

                if disp_res.should_exit:
                    break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting voice assistant...")
            break


def main():
    parser = argparse.ArgumentParser(
        description="In-Vehicle AI-Driven Digital Voice Assistant"
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Process a single WAV audio file and output the response",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        help="Process all WAV files in a directory and print performance benchmarks",
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "physical"],
        default=config.hardware_mode,
        help="Hardware execution mode (default: simulation)",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        default=False,
        help="Enable spoken text-to-speech audio responses",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the mobile smartphone Wi-Fi voice bridge web server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for mobile web server (default: 8080)",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="Enable HTTPS on web server using self-signed certificate",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive voice/text prompt loop",
    )
    parser.add_argument(
        "--validate-dataset",
        action="store_true",
        help="Validate canonical dataset schema, splits, and leaks",
    )
    parser.add_argument(
        "--eval-dataset",
        action="store_true",
        help="Evaluate intent classifier accuracy across text dataset",
    )
    parser.add_argument(
        "--eval-stt",
        action="store_true",
        help="Run end-to-end Whisper STT and Intent evaluation benchmark",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test", "all"],
        help="Split to evaluate for --eval-dataset or --eval-stt (default: test)",
    )

    args = parser.parse_args()

    # CLI evaluation utilities
    if args.validate_dataset:
        from scripts.validate_dataset import validate_dataset
        success = validate_dataset()
        sys.exit(0 if success else 1)

    if args.eval_dataset:
        from scripts.eval_intents_text import evaluate_text_dataset
        evaluate_text_dataset("data/stt/commands.jsonl", args.split)
        return

    if args.eval_stt:
        from scripts.evaluate_stt import run_stt_evaluation
        run_stt_evaluation(split=args.split)
        return

    # If web server mode is requested, run web server directly
    if args.web:
        from app.web import run_server
        run_server(port=args.port, mode=args.mode, use_ssl=args.ssl)
        return

    # Enforce safe physical hardware requirements
    if args.mode == "physical":
        print("[SAFETY NOTICE] Physical mode requested. Verifying hardware permissions...")

    hardware = get_hardware(args.mode)
    transcriber = WhisperTranscriber(config)
    classifier = IntentClassifier()
    dispatcher = CommandDispatcher(hardware, config)
    tts = TextToSpeech(enabled=args.tts)

    try:
        if args.audio:
            if not args.audio.is_file():
                print(f"Error: Audio file not found at '{args.audio}'", file=sys.stderr)
                sys.exit(1)
            result = run_pipeline(args.audio, transcriber, classifier, dispatcher, tts)
            print_result_card(result)

        elif args.batch:
            if not args.batch.is_dir():
                print(f"Error: Directory not found at '{args.batch}'", file=sys.stderr)
                sys.exit(1)

            wav_files = sorted(list(args.batch.glob("*.wav")))
            if not wav_files:
                print(f"No .wav files found in '{args.batch}'")
                return

            print(f"\nProcessing {len(wav_files)} audio files in '{args.batch}' (Mode: {args.mode})...\n")
            total_stt_time = 0.0
            total_total_time = 0.0
            successful = 0

            for wav in wav_files:
                res = run_pipeline(wav, transcriber, classifier, dispatcher, tts)
                print_result_card(res)
                if res["success"]:
                    successful += 1
                total_stt_time += res.get("stt_latency_sec", 0.0)
                total_total_time += res.get("total_latency_sec", 0.0)

            n = len(wav_files)
            print("\n" + "#" * 60)
            print(f" BATCH BENCHMARK SUMMARY ({n} audio samples)")
            print(f" Success Rate          : {successful}/{n} ({successful/n * 100:.1f}%)")
            print(f" Avg STT Latency       : {total_stt_time / n:.3f} s")
            print(f" Avg End-to-End Latency: {total_total_time / n:.3f} s")
            print("#" * 60 + "\n")

        else:
            interactive_loop(transcriber, classifier, dispatcher, tts)

    finally:
        hardware.cleanup()


if __name__ == "__main__":
    main()
