"""Audio input validation and processing utilities."""

import logging
import wave
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def inspect_wav_header(file_path: Path | str) -> Optional[Dict[str, Any]]:
    """Inspects a WAV file header using Python's standard wave library.

    Returns:
        Dictionary with sample rate, channels, bit depth, frames, and duration,
        or None if not a valid WAV file.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        return None

    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / float(framerate) if framerate > 0 else 0.0

            return {
                "channels": channels,
                "bit_depth": sample_width * 8,
                "sample_rate": framerate,
                "frames": n_frames,
                "duration_sec": round(duration, 3),
                "is_16khz": (framerate == 16000),
                "is_mono": (channels == 1),
            }
    except Exception as e:
        logger.warning(f"Could not parse WAV header for '{path}': {e}")
        return None
