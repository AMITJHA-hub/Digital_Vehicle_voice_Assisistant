"""Lightweight offline Text-to-Speech (TTS) engine using espeak-ng."""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Provides offline speech synthesis with robust error handling and fallback."""

    def __init__(self, speed_wpm: int = 155, pitch: int = 50, enabled: bool = True):
        self.speed_wpm = speed_wpm
        self.pitch = pitch
        self.enabled = enabled
        self._espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self._espeak_bin:
            logger.warning("Neither espeak-ng nor espeak found on PATH. Audio playback will be disabled.")

    def synthesize_to_wav(self, text: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Synthesizes text into a WAV audio file.

        Args:
            text: Text to speak.
            output_path: Optional destination Path. If None, creates a temporary file.

        Returns:
            Path to the generated WAV file, or None if synthesis failed.
        """
        if not self._espeak_bin or not text.strip():
            return None

        clean_text = self._prepare_for_speech(text)
        if not clean_text:
            return None

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            dest_path = Path(tmp.name)
        else:
            dest_path = Path(output_path).resolve()
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._espeak_bin,
            "-s", str(self.speed_wpm),
            "-p", str(self.pitch),
            "-w", str(dest_path),
            clean_text,
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and dest_path.is_file():
                return dest_path
            else:
                logger.warning(f"TTS synthesis failed: {res.stderr}")
                return None
        except Exception as e:
            logger.exception(f"Error during TTS synthesis: {e}")
            return None

    def speak(self, text: str, play_audio: bool = True) -> Optional[Path]:
        """Synthesizes and optionally plays spoken audio.

        Args:
            text: Text to speak.
            play_audio: If True, plays through available system audio output.

        Returns:
            Path to synthesized WAV file (if created), or None.
        """
        clean_text = self._prepare_for_speech(text)
        if not clean_text or not self.enabled:
            return None

        # Always synthesize WAV (useful for local inspection, unit tests, and phone Wi-Fi streaming)
        wav_path = self.synthesize_to_wav(clean_text)

        if play_audio and self._espeak_bin:
            try:
                # Play audio via espeak-ng directly
                cmd = [
                    self._espeak_bin,
                    "-s", str(self.speed_wpm),
                    "-p", str(self.pitch),
                    clean_text,
                ]
                subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except Exception as e:
                logger.warning(f"Audio playback skipped or failed: {e}")

        return wav_path

    @staticmethod
    def _prepare_for_speech(text: str) -> str:
        """Removes simulation labels, special brackets, and formatting before speaking."""
        # Remove [Simulated] or other bracketed tags so they aren't spoken aloud
        cleaned = re.sub(r"\[.*?\]", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
