import json
import logging
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

from ..config import config, AppConfig
from ..hardware import get_hardware
from ..hardware.interface import AbstractVehicleHardware
from ..intents import IntentClassifier
from ..assistant import CommandDispatcher
from ..speech import WhisperTranscriber
from ..audio import TextToSpeech

logger = logging.getLogger(__name__)

# Global instances for the request handler
_hardware: Optional[AbstractVehicleHardware] = None
_transcriber: Optional[WhisperTranscriber] = None
_classifier: Optional[IntentClassifier] = None
_dispatcher: Optional[CommandDispatcher] = None
_tts: Optional[TextToSpeech] = None
_latest_response_audio: Optional[Path] = None
_server_mode: str = "simulation"


class CockpitHandler(BaseHTTPRequestHandler):
    """Handles mobile voice uploads, telemetry polling, and static assets."""

    def log_message(self, format, *args):
        # Clean logging
        logger.info(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        global _hardware, _latest_response_audio

        if self.path in ("/", "/index.html"):
            index_path = Path(__file__).resolve().parent / "static" / "index.html"
            if index_path.is_file():
                content = index_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "index.html not found")

        elif self.path == "/api/status":
            status_payload = {
                "lights": _hardware.get_lights_status() if _hardware else False,
                "ac": _hardware.get_ac_status() if _hardware else False,
                "emergency": _hardware.get_emergency_status() if _hardware else False,
                "doors": _hardware.get_door_status() if _hardware else {"driver": "closed", "passenger": "closed"},
                "distance": _hardware.get_distance().value if _hardware else None,
                "temperature": _hardware.get_temperature().value if _hardware else None,
                "humidity": _hardware.get_humidity().value if _hardware else None,
                "ir_detected": getattr(_hardware, "ir_detected", False),
                "mode": _server_mode,
            }
            res_bytes = json.dumps(status_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res_bytes)))
            self.end_headers()
            self.wfile.write(res_bytes)

        elif self.path.startswith("/api/response_audio"):
            if _latest_response_audio and _latest_response_audio.is_file():
                audio_bytes = _latest_response_audio.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(audio_bytes)))
                self.end_headers()
                self.wfile.write(audio_bytes)
            else:
                self.send_error(404, "No response audio available")

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        global _hardware, _transcriber, _classifier, _dispatcher, _tts, _latest_response_audio

        if self.path == "/api/emergency":
            if _hardware:
                current = _hardware.get_emergency_status()
                new_state = not current
                _hardware.set_emergency(new_state)
                res_bytes = json.dumps({"emergency": new_state}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_bytes)))
                self.end_headers()
                self.wfile.write(res_bytes)
            else:
                self.send_error(500, "Hardware not initialized")
            return

        if self.path == "/api/control":
            if not _hardware:
                self.send_error(500, "Hardware not initialized")
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                data = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                data = {}

            action = data.get("action", "")
            success = True
            msg = "OK"

            if action == "lights":
                state = bool(data.get("state", False))
                resp = _hardware.set_lights(state)
                success = resp.success
                msg = resp.message

            elif action == "ac":
                state = bool(data.get("state", False))
                resp = _hardware.set_ac(state)
                success = resp.success
                msg = resp.message

            elif action == "door":
                door = data.get("door", "both")
                state = data.get("state", "open")
                if state == "open":
                    resp = _hardware.open_door(door)
                else:
                    resp = _hardware.close_door(door)
                success = resp.success
                msg = resp.message

            elif action == "emergency":
                if "state" in data:
                    state = bool(data["state"])
                else:
                    state = not _hardware.get_emergency_status()
                resp = _hardware.set_emergency(state)
                success = resp.success
                msg = resp.message

            elif action == "sensors":
                msg = "Sensors refreshed"

            payload = {
                "success": success,
                "message": msg,
                "lights": _hardware.get_lights_status(),
                "ac": _hardware.get_ac_status(),
                "emergency": _hardware.get_emergency_status(),
                "doors": _hardware.get_door_status(),
            }
            res_bytes = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res_bytes)))
            self.end_headers()
            self.wfile.write(res_bytes)
            return

        if self.path == "/api/execute_text":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                data = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                data = {}

            text = data.get("text", "").strip()
            if not text:
                err_bytes = json.dumps({"error": "Empty command text"}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
                return

            t_start = time.perf_counter()
            cmd = _classifier.classify(text)
            disp_res = _dispatcher.dispatch(cmd)

            has_audio = False
            if _tts and _tts.enabled:
                resp_wav = _tts.speak(disp_res.response_text, play_audio=True)
                if resp_wav and resp_wav.is_file():
                    _latest_response_audio = resp_wav
                    has_audio = True

            total_latency = time.perf_counter() - t_start

            payload = {
                "success": disp_res.success,
                "transcription": text,
                "intent": cmd.intent.value,
                "entities": cmd.entities,
                "actions": [a.to_dict() for a in cmd.actions],
                "response": disp_res.response_text,
                "total_latency_sec": round(total_latency, 3),
                "has_audio": has_audio,
            }
            res_bytes = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res_bytes)))
            self.end_headers()
            self.wfile.write(res_bytes)
            return

        if self.path == "/api/command":
            t_start = time.perf_counter()

            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))

            body_bytes = self.rfile.read(content_length)

            if "multipart/form-data" in content_type and "boundary=" in content_type:
                boundary = content_type.split("boundary=")[-1].strip()
                if boundary.startswith('"') and boundary.endswith('"'):
                    boundary = boundary[1:-1]
                boundary_marker = f"--{boundary}".encode()
                parts = body_bytes.split(boundary_marker)
                raw_audio_bytes = b""
                for part in parts:
                    if b'filename=' in part or b'name="audio"' in part:
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            content = part[header_end + 4 :]
                            if content.endswith(b"\r\n"):
                                content = content[:-2]
                            raw_audio_bytes = content
                            break
                if not raw_audio_bytes:
                    raw_audio_bytes = body_bytes
            else:
                raw_audio_bytes = body_bytes

            if not raw_audio_bytes:
                err_bytes = json.dumps({"error": "No audio payload received"}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
                return

            # Save uploaded raw audio to a temp file
            with tempfile.NamedTemporaryFile(suffix=".upload", delete=False) as raw_tmp:
                raw_tmp.write(raw_audio_bytes)
                raw_upload_path = Path(raw_tmp.name)

            # Transcode with ffmpeg to 16 kHz 16-bit mono WAV (optimal for whisper.cpp)
            converted_wav = raw_upload_path.with_suffix(".16k.wav")
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", str(raw_upload_path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(converted_wav)
            ]
            conv_proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            # Clean raw upload
            raw_upload_path.unlink(missing_ok=True)

            if conv_proc.returncode != 0 or not converted_wav.is_file():
                err_msg = f"Audio transcoding failed: {conv_proc.stderr[:200]}"
                err_bytes = json.dumps({"error": err_msg}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
                return

            try:
                # 1. Speech Recognition
                stt_res = _transcriber.transcribe(converted_wav)

                # 2. Intent Classification
                cmd = _classifier.classify(stt_res.text)

                # 3. Hardware Dispatch
                disp_res = _dispatcher.dispatch(cmd)

                # 4. Spoken Feedback & Response Audio Generation
                has_audio = False
                if _tts and _tts.enabled:
                    resp_wav = _tts.speak(disp_res.response_text, play_audio=True)
                    if resp_wav and resp_wav.is_file():
                        _latest_response_audio = resp_wav
                        has_audio = True

                total_latency = time.perf_counter() - t_start

                payload = {
                    "success": disp_res.success,
                    "transcription": stt_res.text,
                    "intent": cmd.intent.value,
                    "entities": cmd.entities,
                    "actions": [a.to_dict() for a in cmd.actions],
                    "response": disp_res.response_text,
                    "stt_latency_sec": stt_res.latency_sec,
                    "total_latency_sec": round(total_latency, 3),
                    "has_audio": has_audio,
                }

                res_bytes = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_bytes)))
                self.end_headers()
                self.wfile.write(res_bytes)

            finally:
                converted_wav.unlink(missing_ok=True)

        else:
            self.send_error(404, "Not Found")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    mode: str = "physical",
    use_ssl: bool = False,
):
    global _hardware, _transcriber, _classifier, _dispatcher, _tts, _server_mode

    _server_mode = mode
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    print("=" * 70)
    print("      IN-VEHICLE DIGITAL COCKPIT - MOBILE WI-FI AUDIO BRIDGE")
    print("=" * 70)

    # Initialize subsystems
    _hardware = get_hardware(mode)
    _transcriber = WhisperTranscriber(config)
    _classifier = IntentClassifier()
    _dispatcher = CommandDispatcher(_hardware, config)
    _tts = TextToSpeech(enabled=True)

    http_server = HTTPServer((host, port), CockpitHandler)
    https_server = None

    cert_file = config.base_dir / "app" / "cert" / "cert.pem"
    key_file = config.base_dir / "app" / "cert" / "key.pem"
    ssl_port = 8443

    if cert_file.is_file() and key_file.is_file():
        try:
            https_server = HTTPServer((host, ssl_port), CockpitHandler)
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
            https_server.socket = ssl_context.wrap_socket(https_server.socket, server_side=True)
            # Run HTTPS in background thread
            https_thread = threading.Thread(target=https_server.serve_forever, daemon=True)
            https_thread.start()
        except Exception as e:
            logger.warning(f"Could not bind HTTPS port {ssl_port}: {e}")
            https_server = None

    ip_cmd = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
    local_ip = ip_cmd.stdout.split()[0] if ip_cmd.stdout else "localhost"

    print(f"\n Cockpit Voice Bridge Server Running:")
    print(f" [1] Standard HTTP : http://{local_ip}:{port}")
    if https_server:
        print(f" [2] Secure HTTPS   : https://{local_ip}:{ssl_port}  (Unlocks Live In-Browser Mic!)")
    print(f" -> Hardware Mode   : {mode.upper()}")
    print("\n Instructions for Mobile Smartphone:")
    print(" 1. Ensure phone is on the same Wi-Fi / Hotspot.")
    if https_server:
        print(f" 2. For LIVE in-browser mic: Open Chrome/Safari to: https://{local_ip}:{ssl_port}")
        print("    (Tap 'Advanced' -> 'Proceed to {local_ip}' to accept self-signed SSL cert).")
    print(f" 3. For standard access: Open http://{local_ip}:{port}")
    print(" 4. Use voice commands to test Headlights, AC, Proximity, and Emergency Mode.")
    print(" Press Ctrl+C to stop server.\n" + "=" * 70 + "\n")

    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down cockpit server...")
    finally:
        http_server.server_close()
        if https_server:
            https_server.server_close()
        if _hardware:
            _hardware.cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cockpit Mobile Voice Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--mode", choices=["simulation", "physical"], default="physical", help="Hardware mode")
    parser.add_argument("--ssl", action="store_true", help="Enable HTTPS using self-signed certificate")
    args = parser.parse_args()

    run_server(port=args.port, mode=args.mode, use_ssl=args.ssl)
