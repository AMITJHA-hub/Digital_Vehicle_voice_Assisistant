"""Speech processing module integrating whisper.cpp."""

from .transcriber import WhisperTranscriber, TranscriptionResult

__all__ = ["WhisperTranscriber", "TranscriptionResult"]
