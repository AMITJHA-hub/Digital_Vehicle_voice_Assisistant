"""Unit and integration tests for the mobile web bridge server."""

import json
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from app.config import config
from app.hardware.simulation import SimulatedVehicleHardware
from app.intents import IntentClassifier
from app.assistant import CommandDispatcher
from app.speech import WhisperTranscriber
from app.audio import TextToSpeech
from app.web import server as web_server


class TestWebServer(unittest.TestCase):
    """Verifies HTTP endpoints for phone connectivity and audio upload."""

    @classmethod
    def setUpClass(cls):
        # Configure test server on a local ephemeral port
        cls.port = 8899
        cls.server_address = f"http://127.0.0.1:{cls.port}"

        web_server._hardware = SimulatedVehicleHardware()
        web_server._transcriber = WhisperTranscriber(config)
        web_server._classifier = IntentClassifier()
        web_server._dispatcher = CommandDispatcher(web_server._hardware, config)
        web_server._tts = TextToSpeech(enabled=False)

        cls.httpd = web_server.HTTPServer(("127.0.0.1", cls.port), web_server.CockpitHandler)
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_get_root_serves_html(self):
        """Verifies GET / returns the cockpit web interface."""
        req = urllib.request.Request(f"{self.server_address}/")
        with urllib.request.urlopen(req, timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            content_type = resp.headers.get("Content-Type", "")
            self.assertIn("text/html", content_type)
            body = resp.read().decode("utf-8")
            self.assertIn("Vehicle Cockpit", body)

    def test_get_api_status(self):
        """Verifies GET /api/status returns valid JSON telemetry."""
        req = urllib.request.Request(f"{self.server_address}/api/status")
        with urllib.request.urlopen(req, timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("lights", data)
            self.assertIn("ac", data)
            self.assertIn("distance", data)
            self.assertIn("temperature", data)

    def test_post_audio_command(self):
        """Verifies POST /api/command accepts audio and returns transcribed command response."""
        sample_audio = config.base_dir / "app" / "audio_samples" / "lights_on.wav"
        self.assertTrue(sample_audio.is_file(), "Sample audio file missing")

        audio_bytes = sample_audio.read_bytes()
        req = urllib.request.Request(
            f"{self.server_address}/api/command",
            data=audio_bytes,
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            result = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(result.get("success"))
            self.assertIn("Turn on the lights", result.get("transcription", ""))
            self.assertEqual(result.get("intent"), "LIGHTS_ON")
            self.assertIn("headlights turned on", result.get("response", "").lower())

    def test_post_control_endpoint(self):
        """Verifies POST /api/control triggers hardware actions directly."""
        req_data = json.dumps({"action": "lights", "state": True}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_address}/api/control",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            result = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(result.get("success"))
            self.assertTrue(result.get("lights"))

    def test_post_execute_text_endpoint(self):
        """Verifies POST /api/execute_text executes compound text commands."""
        req_data = json.dumps({"text": "turn on headlights and turn off ac"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_address}/api/execute_text",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            self.assertEqual(resp.status, 200)
            result = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(result.get("success"))
            self.assertEqual(result.get("intent"), "COMPOUND")
            self.assertEqual(len(result.get("actions", [])), 2)
            self.assertIn("headlights turned on", result.get("response", "").lower())


if __name__ == "__main__":
    unittest.main()
