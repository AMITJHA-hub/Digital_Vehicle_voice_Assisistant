"""Generates benchmark voice command WAV audio files using espeak-ng."""

import subprocess
import sys
from pathlib import Path

SAMPLE_COMMANDS = {
    "hello.wav": "Hello assistant",
    "cabin_temp.wav": "What is the cabin temperature?",
    "humidity.wav": "What is the humidity?",
    "obstacle.wav": "How far is the obstacle?",
    "sensor_status.wav": "Check the vehicle sensors.",
    "lights_on.wav": "Turn on the lights.",
    "lights_off.wav": "Turn off the lights.",
    "ac_on.wav": "Turn the AC on.",
    "ac_off.wav": "Turn the AC off.",
    "ac_status.wav": "What is the AC status?",
    "open_driver_door.wav": "Open the driver door.",
    "open_door_unspecified.wav": "Open the door.",
    "close_door.wav": "Close the door.",
    "help.wav": "Help me please.",
    "exit.wav": "Exit the application.",
}


def generate_samples(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(SAMPLE_COMMANDS)} sample audio command files into {output_dir}...")

    for filename, phrase in SAMPLE_COMMANDS.items():
        dest = output_dir / filename
        cmd = ["espeak-ng", "-w", str(dest), phrase]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  [OK] {filename} -> \"{phrase}\"")
        else:
            print(f"  [FAIL] {filename}: {res.stderr}")


if __name__ == "__main__":
    target_dir = Path(__file__).resolve().parent.parent / "audio_samples"
    generate_samples(target_dir)
