# In-Vehicle AI Digital Voice Assistant: STT Evaluation Dataset

## 1. Overview and Purpose

This directory contains the canonical evaluation dataset and benchmark suite for the offline in-vehicle digital voice assistant deployed on the Raspberry Pi 5 platform. The dataset is specifically engineered to benchmark Speech-to-Text (STT) accuracy with `whisper.cpp` (`tiny.en`), evaluate deterministic intent parsing, verify complex compound voice command pipelines, and ensure rigorous fail-safe rejection behavior for out-of-scope or ambiguous requests.

The entire dataset and evaluation pipeline operates 100% locally and offline without external cloud dependencies or LLM overhead.

---

## 2. Directory Structure

```text
data/stt/
├── README.md               # Complete dataset specification and documentation
├── commands.jsonl          # Canonical machine-readable dataset (one JSON per line)
├── commands.csv            # Human-readable spreadsheet export
├── eval_report.json        # Empirical STT and intent benchmark results
└── audio/                  # Generated 16 kHz 16-bit mono WAV audio clips
```

---

## 3. Dataset Schema

Every record in `commands.jsonl` adheres to the following strict JSON schema:

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `id` | `string` | Unique sequential identifier (`CMD_xxx`) | `"CMD_014"` |
| `utterance` | `string` | Natural spoken utterance including punctuation | `"turn off the ac and headlights"` |
| `normalized_utterance` | `string` | Lowercase, contraction-expanded, punctuation-stripped canonical form | `"turn off the ac and headlights"` |
| `expected_actions` | `array[object]` | Ordered list of typed actions to execute | `[{"type": "TURN_AC_OFF", "parameters": {}}, {"type": "LIGHTS_OFF", "parameters": {}}]` |
| `category` | `string` | Domain category | `"compound"` |
| `split` | `string` | Deterministic evaluation split (`train`, `dev`, `test`) | `"train"` |
| `difficulty` | `string` | Linguistic difficulty (`easy`, `medium`, `hard`) | `"medium"` |
| `supported` | `boolean` | `true` if fully actionable; `false` if unsupported, out-of-scope, or rejected | `true` |
| `notes` | `string` | Rationale, linguistic phenomenon, or failure explanation | `"Compound object turn off AC and lights"` |

### Action Object Schema

Each element inside `expected_actions` contains:
```json
{
  "type": "INTENT_TYPE_ENUM",
  "parameters": {
    "key": "value"
  }
}
```

---

## 4. Intent & Action Vocabulary

The system recognizes 18 atomic vehicle intents and queries:

| Intent Type | Category | Description | Parameters |
| :--- | :--- | :--- | :--- |
| `LIGHTS_ON` | Actuator | Turn on vehicle headlights | None (`{}`) |
| `LIGHTS_OFF` | Actuator | Turn off vehicle headlights | None (`{}`) |
| `GET_LIGHTS_STATUS` | Query | Query whether headlights are on or off | None (`{}`) |
| `TURN_AC_ON` | Actuator | Turn on air conditioning / cooling | None (`{}`) |
| `TURN_AC_OFF` | Actuator | Turn off air conditioning / cooling | None (`{}`) |
| `GET_AC_STATUS` | Query | Query whether AC is currently active | None (`{}`) |
| `OPEN_DOOR` | Actuator | Open vehicle servo door | `door`: `"driver"`, `"passenger"`, `"both"`, `"unspecified"` |
| `CLOSE_DOOR` | Actuator | Close vehicle servo door | `door`: `"driver"`, `"passenger"`, `"both"`, `"unspecified"` |
| `GET_DOOR_STATUS` | Query | Query status of vehicle doors | `door`: `"driver"`, `"passenger"`, `"both"` (optional) |
| `GET_TEMPERATURE` | Sensor | Read cabin temperature via DHT22 sensor | None (`{}`) |
| `GET_HUMIDITY` | Sensor | Read cabin relative humidity via DHT22 sensor | None (`{}`) |
| `GET_DISTANCE` | Sensor | Read front obstacle distance via HC-SR04 ultrasonic sensor | None (`{}`) |
| `GET_SENSOR_STATUS` | Sensor | Full diagnostic scan of all environmental sensors | None (`{}`) |
| `EMERGENCY_ON` | Safety | Enable hazard emergency warning flashers | None (`{}`) |
| `EMERGENCY_OFF` | Safety | Disable hazard emergency warning flashers | None (`{}`) |
| `GREETING` | Assistant | Conversational greetings ("hello", "good morning") | None (`{}`) |
| `HELP` | Assistant | List available vehicle voice assistant capabilities | None (`{}`) |
| `STOP_LISTENING` | Session | Pause listening mode / mute assistant | None (`{}`) |
| `EXIT` | Session | Cleanly shutdown voice assistant software | None (`{}`) |

---

## 5. Compound Voice Command Pipeline

The assistant implements deterministic compound sentence processing supporting multi-action workflows:

### A. Compound Direct Objects (Single Distributed Verb)
When a single verb applies to multiple targets:
- *"Turn on the AC and headlights"* $\rightarrow$ `[TURN_AC_ON, LIGHTS_ON]`
- *"Turn off both the AC and the lights"* $\rightarrow$ `[TURN_AC_OFF, LIGHTS_OFF]`
- *"Activate the headlights and air conditioning"* $\rightarrow$ `[LIGHTS_ON, TURN_AC_ON]`

### B. Sequential Clauses and Markers
Sequential ordering using conjunctions and time adverbs:
- *"Switch off the lights and turn off the AC"* $\rightarrow$ `[LIGHTS_OFF, TURN_AC_OFF]`
- *"Turn off the AC first, then turn off the headlights"* $\rightarrow$ `[TURN_AC_OFF, LIGHTS_OFF]`
- *"Open the driver door and turn on the headlights"* $\rightarrow$ `[OPEN_DOOR(driver), LIGHTS_ON]`

### C. Sequential Inversion (`before`)
Commands structured with *"before [A], [B]"* execute $[B]$ first, then $[A]$:
- *"Before opening the passenger door, turn off the AC"* $\rightarrow$ `[TURN_AC_OFF, OPEN_DOOR(passenger)]`

### D. Anaphoric Pronoun Resolution
Resolves pronouns ("them", "it") referencing devices mentioned in preceding clauses:
- *"Turn on the headlights, then turn them off"* $\rightarrow$ `[LIGHTS_ON, LIGHTS_OFF]`
- *"Start the AC, then turn it off"* $\rightarrow$ `[TURN_AC_ON, TURN_AC_OFF]`

### E. Contrast and State Preservation (`keep`, `leave`)
Contrast phrases that maintain or enforce states:
- *"Turn off the headlights but keep the AC on"* $\rightarrow$ `[LIGHTS_OFF, TURN_AC_ON]`
- *"Keep the lights on but turn the AC off"* $\rightarrow$ `[LIGHTS_ON, TURN_AC_OFF]`

### F. Negative Contrast
Suppresses the negated target while executing the affirmative clause:
- *"Turn off the AC, not the headlights"* $\rightarrow$ `[TURN_AC_OFF]`

### G. Three-Action Sequences
- *"Turn on the headlights, AC, and open the driver door"* $\rightarrow$ `[LIGHTS_ON, TURN_AC_ON, OPEN_DOOR(driver)]`
- *"Turn off the AC, switch off the headlights, and close both doors"* $\rightarrow$ `[TURN_AC_OFF, LIGHTS_OFF, CLOSE_DOOR(both)]`

---

## 6. Safety Rejection and Ambiguity Policy

To prevent unintended vehicle actuation and protect passenger safety:

1. **Mixed Supported / Unsupported Rejection**: If a compound command contains any unsupported action (e.g. *"turn off the AC and play music"* or *"open driver door and navigate home"*), the **entire command is rejected** (`CommandStatus.REJECTED`). No partial action is executed.
2. **Ambiguous Door Target in Compound**: In a compound command, an unspecified door (e.g. *"open the door and turn on the AC"*) is rejected rather than arbitrarily selecting a door.
3. **Standalone Ambiguous Door Handling**: Standalone commands like *"open the door"* default safely to the driver door while verbally informing the driver: *"Opening driver doors by default. You can also specify driver or passenger door."*
4. **Out-of-Scope Hardware**: Commands attempting to actuate nonexistent physical locks (*"unlock the door"*), windows, radios, or temperature values (*"set temperature to 20"*) are rejected (`CommandStatus.UNSUPPORTED`).

---

## 7. Dataset Statistics and Splits

The dataset comprises **310 canonical utterances** with zero duplicate leaks across splits:

### Split Distribution
| Split | Count | Percentage | Minimum Target |
| :--- | :--- | :--- | :--- |
| `train` | 183 | 59.0% | ~60% |
| `dev` | 59 | 19.0% | ~20% |
| `test` | 68 | 21.9% | $\ge 50$ (Held-out) |
| **Total** | **310** | **100.0%** | **250–350** |

### Category Distribution
| Category | Count | Percentage |
| :--- | :--- | :--- |
| `compound` | 72 | 23.2% |
| `doors` | 42 | 13.5% |
| `lighting` | 36 | 11.6% |
| `climate` | 36 | 11.6% |
| `sensors` | 36 | 11.6% |
| `unsupported` | 40 | 12.9% |
| `assistant` | 30 | 9.7% |
| `emergency` | 18 | 5.8% |
| **Total** | **310** | **100.0%** |

---

## 8. CLI Tools and Evaluation Scripts

All scripts are located in `scripts/`:

1. **Dataset Validation**:
   ```bash
   python scripts/validate_dataset.py
   ```
   Validates JSON schema, unique IDs, category definitions, split integrity, and leakage.

2. **CSV Export**:
   ```bash
   python scripts/export_csv.py
   ```
   Exports `commands.jsonl` into `commands.csv`.

3. **Text Intent Classification Benchmark**:
   ```bash
   python scripts/eval_intents_text.py
   ```
   Evaluates the deterministic classifier against text utterances in `commands.jsonl` across all splits and categories.

4. **Synthetic Audio Generation**:
   ```bash
   python scripts/generate_stt_audio.py --split test
   ```
   Generates standard 16 kHz 16-bit mono WAV files using `espeak-ng` and `ffmpeg` in `data/stt/audio/`.

5. **End-to-End STT + Intent Evaluation**:
   ```bash
   python scripts/evaluate_stt.py --split test
   ```
   Transcribes audio using `whisper.cpp`, parses the resulting transcripts, calculates Word Error Rate (WER), Transcript Exact Match, and Action Sequence Exact Match, and outputs `data/stt/eval_report.json`.

---

## 9. Instructions for Adding New Commands

To contribute new utterances to the dataset:
1. Append your record to `raw_data` in `scripts/generate_dataset.py`.
2. Ensure you assign a unique category, deterministic split (`train`, `dev`, or `test`), difficulty level, and machine-readable `expected_actions`.
3. If introducing a new intent, update `IntentType` in `app/intents/models.py` and implement the corresponding regex in `app/intents/classifier.py`.
4. Run `python scripts/generate_dataset.py` to regenerate `commands.jsonl` and `commands.csv`.
5. Run `python scripts/validate_dataset.py` and `python scripts/eval_intents_text.py` to confirm 100% test validity.
