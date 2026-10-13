# In-Vehicle AI-Driven Digital Voice Assistant
## Engineering Session Notes & Project Progress

**Date**: 2026-10-12  
**Target Platform**: Raspberry Pi 5 Model B Rev 1.1 (8 GB RAM)  
**Operating System**: Debian GNU/Linux 13 (Trixie), Kernel 6.12 series, ARM64 (`aarch64`)  
**Python Runtime**: Python 3.13.5 (Virtual Environment at `~/.venv` with system hardware packages enabled)  
**Primary Engine**: `whisper.cpp` (ARM NEON / FP16 / DOTPROD vector accelerated)

---

### 1. Phase Completion Status

| Phase | Description | Status | Validation Summary |
|---|---|---|---|
| **Phase 1** | System & Environment Inspection | **COMPLETED** | Verified Pi 5, RP1 chip, 7.9 GB RAM, 19 GB free disk, no physical capture devices. |
| **Phase 2** | `whisper.cpp` Build & Benchmark | **COMPLETED** | Built `whisper-cli` with CMake Release; downloaded `tiny.en` (77 MB); transcribed `jfk.wav` in 1.57s (RTF 0.14x, ~7x faster than real-time). |
| **Phase 3** | App Architecture & WAV Interface | **COMPLETED** | Created modular `app/` structure; built `WhisperTranscriber` and `inspect_wav_header`. |
| **Phase 4** | Deterministic Offline Intents | **COMPLETED** | Implemented `IntentClassifier` & `CommandDispatcher` covering all 16 intents, door entity extraction, and simulation hardware. |
| **Phase 5** | Simulation Demo & Benchmark | **COMPLETED** | Generated 15 audio command WAVs; full batch pipeline achieved **15/15 (100%)** success at **1.082s average STT latency**. |
| **Phase 6** | Speech Output (TTS) | **COMPLETED** | Integrated `espeak-ng` offline TTS; formats text to strip bracketed tags before speaking; CLI `--tts` flag added. |
| **Phase 7** | RP1 GPIO Physical Hardware | **COMPLETED** | Verified White, Green, Red LEDs; IR obstacle sensor; HC-SR04 ultrasonic sensor with custom voltage divider; DHT22 temperature & humidity sensor decoded with live telemetry (27°C, 59% humidity). Servos safely locked without external power. |
| **Phase 8** | Android Wi-Fi Audio Bridge | **COMPLETED** | Lightweight HTTP server + HTML5 mobile cockpit dashboard running on port 8080. Smartphone connected and streaming live commands. |
| **Phase 9** | Final Demo & Documentation | **COMPLETED** | Full project [`README.md`](file:///home/re/car-voice-assistant/README.md) written with architecture diagrams, wiring pin tables, benchmark data, and CLI usage. |

---

### 2. Comprehensive Manifest of Created & Modified Files

* [`app/__init__.py`](file:///home/re/car-voice-assistant/app/__init__.py): Application package initializer (v1.0.0).
* [`app/config.py`](file:///home/re/car-voice-assistant/app/config.py): Central configuration defining paths, model parameters, and safe simulation-first defaults.
* [`app/main.py`](file:///home/re/car-voice-assistant/app/main.py): Application entry point supporting `--audio`, `--batch`, `--interactive`, `--mode`, `--tts`, and `--web`.
* [`app/speech/transcriber.py`](file:///home/re/car-voice-assistant/app/speech/transcriber.py): `WhisperTranscriber` wrapping `whisper-cli`.
* [`app/speech/audio_input.py`](file:///home/re/car-voice-assistant/app/speech/audio_input.py): WAV header inspector.
* [`app/intents/models.py`](file:///home/re/car-voice-assistant/app/intents/models.py): `IntentType` enum (16 vehicle intents) and `ParsedCommand`.
* [`app/intents/normalizer.py`](file:///home/re/car-voice-assistant/app/intents/normalizer.py): Text normalizer expanding spoken contractions.
* [`app/intents/classifier.py`](file:///home/re/car-voice-assistant/app/intents/classifier.py): Deterministic regex pattern matcher and door entity extractor.
* [`app/assistant/dispatcher.py`](file:///home/re/car-voice-assistant/app/assistant/dispatcher.py): Command dispatcher mapping intents to hardware actions and natural responses.
* [`app/hardware/interface.py`](file:///home/re/car-voice-assistant/app/hardware/interface.py): Abstract hardware contract (`AbstractVehicleHardware`).
* [`app/hardware/simulation.py`](file:///home/re/car-voice-assistant/app/hardware/simulation.py): Simulated hardware layer returning labeled `[Simulated]` telemetry.
* [`app/hardware/physical.py`](file:///home/re/car-voice-assistant/app/hardware/physical.py): Raspberry Pi 5 RP1 GPIO hardware driver using `lgpio`, supporting LEDs, IR sensor, HC-SR04 with pulse timing, and DHT22 single-wire pulse decoding.
* [`app/audio/tts.py`](file:///home/re/car-voice-assistant/app/audio/tts.py): Offline speech synthesizer using `espeak-ng`.
* [`app/web/server.py`](file:///home/re/car-voice-assistant/app/web/server.py): Zero-dependency HTTP/HTTPS cockpit server with live telemetry polling and FFmpeg audio transcoding.
* [`app/web/static/index.html`](file:///home/re/car-voice-assistant/app/web/static/index.html): Mobile-responsive cockpit dashboard with push-to-talk microphone, native recorder fallback, and telemetry gauges.
* [`app/scripts/check_all_hardware.py`](file:///home/re/car-voice-assistant/app/scripts/check_all_hardware.py): Comprehensive hardware diagnostic testing all components without a microphone.
* [`app/scripts/test_hardware.py`](file:///home/re/car-voice-assistant/app/scripts/test_hardware.py): Interactive breadboard diagnostic utility.
* [`app/scripts/generate_sample_audio.py`](file:///home/re/car-voice-assistant/app/scripts/generate_sample_audio.py): Synthesizes 15 benchmark voice command WAVs.
* [`app/tests/`](file:///home/re/car-voice-assistant/app/tests/): 24 unit and integration tests across 5 test suites (100% passing).
* [`README.md`](file:///home/re/car-voice-assistant/README.md): Complete final documentation, Mermaid architecture diagram, wiring guide, and CLI manual.

---

### 3. Server Status & Live Feature Set
* The mobile cockpit server is actively running as a dual-stack daemon:
  * **Standard HTTP**: `http://172.20.10.2:8080` (Native phone sound recorder fallback).
  * **Secure HTTPS**: `https://172.20.10.2:8443` (Unlocks live in-browser push-to-talk mic via `navigator.mediaDevices.getUserMedia`).
* **Active Hardware Features**:
  1. **Live Push-to-Talk**: Live audio recorded directly in mobile browser and transcoded on-the-fly via FFmpeg for `whisper.cpp`.
  2. **AC Climate Indicator**: Green LED (GPIO 27, Pin 13) blinks continuously while AC is ON; turns OFF when AC is OFF (`TURN_AC_ON` / `TURN_AC_OFF`).
  3. **Headlights**: White LED (GPIO 17, Pin 11) solid ON when headlights active; OFF when headlights OFF (`LIGHTS_ON` / `LIGHTS_OFF`).
  4. **IR Obstacle Sensor Indicator**: Red LED (GPIO 22, Pin 15) turns solid ON when an obstacle is within ~4 cm; turns OFF when clear.
  5. **Ultrasonic Proximity Protection**: Yellow LED (GPIO 23, Pin 16) driven by front HC-SR04 distance (re-routed to TRIG GPIO 5 Pin 29, ECHO GPIO 6 Pin 31):
     * `< 5.0 cm`: Yellow LED solid ON.
     * `5.0 cm – 10.0 cm`: Yellow LED rapid blinking.
     * `> 10.0 cm`: Yellow LED OFF.
  6. **Emergency Hazard Mode**: All 4 LEDs (White, Green, Red, Yellow) flash simultaneously in sync (`EMERGENCY_ON`, `EMERGENCY_OFF`, or UI tap).
  7. **Cabin Climate Telemetry**: DHT22 live updates (temperature & humidity) every second.
  8. **SG90 Servo Motors (Doors)**: Powered via external 5V with common Pi ground on Pin 34:
     * **Left (Driver) Door**: Signal on **GPIO 12 (Pin 32)** (Pulsed 2000µs for open, 1000µs for close).
     * **Right (Passenger) Door**: Signal on **GPIO 13 (Pin 33)** (Pulsed 2000µs for open, 1000µs for close).
     * Voice commands: *"Open left door"*, *"Close left door"*, *"Open right door"*, *"Close right door"*, *"Open both doors"*, *"Close both doors"*.
