"""Unit tests for the WhisperTranscriber speech module."""

import unittest
from pathlib import Path

from app.config import config
from app.speech.audio_input import inspect_wav_header
from app.speech.transcriber import WhisperTranscriber


class TestWhisperTranscriber(unittest.TestCase):
    """Verifies offline transcription engine integration."""

    @classmethod
    def setUpClass(cls):
        cls.transcriber = WhisperTranscriber(config)
        cls.jfk_wav = config.base_dir / "whisper.cpp" / "samples" / "jfk.wav"

    def test_sample_wav_header(self):
        """Validates that jfk.wav can be parsed and matches expected WAV format."""
        header = inspect_wav_header(self.jfk_wav)
        self.assertIsNotNone(header, "Failed to read jfk.wav header")
        self.assertEqual(header["sample_rate"], 16000)
        self.assertEqual(header["channels"], 1)
        self.assertGreater(header["duration_sec"], 10.0)

    def test_transcribe_jfk_sample(self):
        """Validates end-to-end transcription using tiny.en on jfk.wav."""
        result = self.transcriber.transcribe(self.jfk_wav)
        self.assertTrue(result.success, f"Transcription failed: {result.error}")
        self.assertIn("americans", result.text.lower())
        self.assertIn("country", result.text.lower())
        self.assertGreater(result.latency_sec, 0.0)
        print(f"\n[Test Result] Latency: {result.latency_sec}s | Transcribed: \"{result.text}\"")

    def test_nonexistent_audio_file(self):
        """Verifies safe error handling when an audio file does not exist."""
        bogus_file = Path("/nonexistent/path/fake.wav")
        result = self.transcriber.transcribe(bogus_file)
        self.assertFalse(result.success)
        self.assertEqual(result.text, "")
        self.assertIn("not found", result.error.lower())

    def test_clean_transcript_artifacts(self):
        """Verifies stripping of brackets, extra spaces, and whisper annotations."""
        raw = " [MUSIC]   Hello World! (applause)   Please open the door.  \n"
        cleaned = self.transcriber._clean_transcript(raw)
        self.assertEqual(cleaned, "Hello World! Please open the door.")


if __name__ == "__main__":
    unittest.main()
