"""Configuration settings for the In-Vehicle Voice Assistant."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration values with safe simulation-first defaults."""

    # Paths
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "VOICE_ASSISTANT_BASE_DIR",
                Path(__file__).resolve().parent.parent,
            )
        )
    )
    app_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )

    # Whisper.cpp engine configuration
    whisper_cli: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "WHISPER_CLI_PATH",
                Path(__file__).resolve().parent.parent
                / "whisper.cpp"
                / "build"
                / "bin"
                / "whisper-cli",
            )
        )
    )
    whisper_model: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "WHISPER_MODEL_PATH",
                Path(__file__).resolve().parent.parent
                / "whisper.cpp"
                / "models"
                / "ggml-tiny.en.bin",
            )
        )
    )
    whisper_threads: int = int(os.environ.get("WHISPER_THREADS", "4"))
    whisper_language: str = "en"
    whisper_timeout_sec: float = 30.0

    # Hardware execution mode: MUST default to "simulation"
    hardware_mode: str = os.environ.get(
        "VOICE_ASSISTANT_HW_MODE", "simulation"
    ).lower()

    # GPIO Pin assignments (BCM numbering scheme, validated for Pi 5 RP1)
    # NOTE: These are defined here for consistency across modules,
    # but NO GPIO access will be made unless hardware_mode == "physical".
    pin_led_headlights: int = 17       # White LED (Headlights / Pin 11)
    pin_led_ac: int = 27               # Green LED (AC Blinking / Pin 13)
    pin_led_warning: int = 22          # Red LED (IR Obstacle alert only / Pin 15)
    pin_led_proximity: int = 23        # Yellow LED (Ultrasonic proximity alert / Pin 16)
    pin_ultrasonic_trig: int = 5       # HC-SR04 TRIG (Output / Re-routed to Pin 29)
    pin_ultrasonic_echo: int = 6       # HC-SR04 ECHO (Input via voltage divider / Re-routed to Pin 31)
    pin_ir_sensor: int = 25            # IR obstacle sensor (Digital Input / Pin 22)
    pin_dht22: int = 4                 # DHT22 / AM2302 (1-wire Data / Pin 7)
    pin_servo_driver_door: int = 12    # SG90 Driver Door (Pin 32 - Disabled by default)
    pin_servo_pass_door: int = 13      # SG90 Passenger Door (Pin 33 - Disabled by default)

    # Default door selection for ambiguous commands
    default_door: str = "both"         # "driver", "passenger", or "both"


# Global singleton instance
config = AppConfig()
