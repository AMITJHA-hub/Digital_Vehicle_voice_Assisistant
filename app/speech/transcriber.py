"""WAV-to-transcript speech recognition interface using whisper.cpp."""

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import config, AppConfig

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Represents the outcome of a speech-to-text transcription."""
    text: str
    latency_sec: float
    success: bool
    error: Optional[str] = None


class WhisperTranscriber:
    """Executes offline speech recognition via the compiled whisper.cpp CLI."""

    def __init__(self, app_config: Optional[AppConfig] = None):
        self.config = app_config or config
        self._verify_engine_available()

    def _verify_engine_available(self) -> None:
        """Verifies that the whisper-cli binary and model file exist on disk."""
        if not self.config.whisper_cli.is_file():
            raise FileNotFoundError(
                f"whisper-cli binary not found at '{self.config.whisper_cli}'. "
                "Ensure whisper.cpp has been built."
            )
        if not os.access(self.config.whisper_cli, os.X_OK):
            raise PermissionError(
                f"whisper-cli at '{self.config.whisper_cli}' is not executable."
            )
        if not self.config.whisper_model.is_file():
            raise FileNotFoundError(
                f"Whisper model file not found at '{self.config.whisper_model}'. "
                "Ensure the model has been downloaded."
            )

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        """Transcribes a WAV audio file into text using whisper.cpp.

        Args:
            audio_path: Path to a 16 kHz 16-bit mono WAV audio file.

        Returns:
            TranscriptionResult containing normalized text, latency, and status.
        """
        path = Path(audio_path).resolve()
        if not path.is_file():
            return TranscriptionResult(
                text="",
                latency_sec=0.0,
                success=False,
                error=f"Audio file not found: {path}",
            )

        cmd = [
            str(self.config.whisper_cli),
            "-m", str(self.config.whisper_model),
            "-f", str(path),
            "-t", str(self.config.whisper_threads),
            "-nt",       # Do not print timestamps
            "-np",       # Do not print progress
            "-l", self.config.whisper_language,
        ]

        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.whisper_timeout_sec,
                check=False,
            )
            latency = time.perf_counter() - start_time

            if result.returncode != 0:
                err_msg = (
                    f"whisper-cli failed (exit code {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
                logger.error(err_msg)
                return TranscriptionResult(
                    text="",
                    latency_sec=latency,
                    success=False,
                    error=err_msg,
                )

            # Clean and normalize raw output
            cleaned_text = self._clean_transcript(result.stdout)
            return TranscriptionResult(
                text=cleaned_text,
                latency_sec=round(latency, 3),
                success=True,
                error=None,
            )

        except subprocess.TimeoutExpired:
            latency = time.perf_counter() - start_time
            err_msg = f"whisper-cli timed out after {self.config.whisper_timeout_sec}s"
            logger.error(err_msg)
            return TranscriptionResult(
                text="",
                latency_sec=latency,
                success=False,
                error=err_msg,
            )
        except Exception as e:
            latency = time.perf_counter() - start_time
            err_msg = f"Unexpected error during transcription: {e}"
            logger.exception(err_msg)
            return TranscriptionResult(
                text="",
                latency_sec=latency,
                success=False,
                error=err_msg,
            )

    @staticmethod
    def _clean_transcript(raw_text: str) -> str:
        """Strips whisper formatting artifacts, punctuation, and extra whitespace."""
        if not raw_text:
            return ""

        # Remove bracketed tokens like [BLANK_AUDIO], (music), [applause]
        cleaned = re.sub(r"\[.*?\]|\(.*?\)", "", raw_text)
        # Collapse multiple whitespaces and trim
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
