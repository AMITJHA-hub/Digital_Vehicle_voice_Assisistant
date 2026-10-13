"""Unit tests for the offline TextToSpeech module."""

import unittest
from pathlib import Path

from app.audio.tts import TextToSpeech
from app.speech.audio_input import inspect_wav_header


class TestTextToSpeech(unittest.TestCase):
    """Verifies offline TTS speech synthesis."""

    def setUp(self):
        self.tts = TextToSpeech(speed_wpm=150, pitch=50, enabled=True)

    def test_prepare_for_speech_strips_brackets(self):
        """Verifies bracketed tags like [Simulated] are stripped before speaking."""
        raw = "[Simulated] Cabin temperature is 23.5 degrees Celsius."
        cleaned = self.tts._prepare_for_speech(raw)
        self.assertEqual(cleaned, "Cabin temperature is 23.5 degrees Celsius.")

    def test_synthesize_to_wav(self):
        """Verifies synthesis generates a valid, playable WAV file."""
        test_wav = self.tts.synthesize_to_wav("Vehicle systems online.")
        self.assertIsNotNone(test_wav)
        self.assertTrue(test_wav.is_file())

        try:
            # Check WAV header
            header = inspect_wav_header(test_wav)
            self.assertIsNotNone(header)
            self.assertGreater(header["duration_sec"], 0.2)
        finally:
            if test_wav.is_file():
                test_wav.unlink()

    def test_tts_disabled_mode(self):
        """Verifies disabled TTS returns None cleanly without error."""
        disabled_tts = TextToSpeech(enabled=False)
        result = disabled_tts.speak("Testing disabled mode.", play_audio=False)
        self.assertIsNone(result)

    def test_empty_text_handling(self):
        """Verifies passing empty or whitespace text is safely handled."""
        result = self.tts.synthesize_to_wav("    ")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
