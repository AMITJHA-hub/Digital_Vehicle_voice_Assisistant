"""Deterministic rule-based compound intent classifier for vehicle voice commands."""

import re
from typing import List, Tuple, Optional, Dict, Any, Pattern

from .models import IntentType, CommandStatus, Action, ParsedCommand
from .normalizer import normalize_text, strip_polite_fillers


class IntentClassifier:
    """Classifies natural language utterances into predefined intents and ordered action lists.

    Operates deterministically and offline without relying on cloud APIs or LLMs.
    Supports atomic single intents, queries, and compound voice commands.
    """

    def __init__(self):
        self._rules: List[Tuple[IntentType, List[Pattern]]] = self._build_rules()

    def _build_rules(self) -> List[Tuple[IntentType, List[Pattern]]]:
        """Compiles prioritized regex patterns for deterministic intent recognition."""
        raw_rules = [
            # Session control / Stop / Exit
            (
                IntentType.STOP_LISTENING,
                [
                    r"\bstop listening\b",
                    r"\bmute assistant\b",
                    r"\bpause (the )?(voice )?assistant\b",
                    r"\bpause listening\b",
                    r"\bgo to sleep\b",
                ],
            ),
            (
                IntentType.EXIT,
                [
                    r"\b(exit|quit|terminate|shutdown|goodbye|bye)\b",
                    r"\b(shut down|close)\b.*?\b(application|assistant|program|system)\b",
                ],
            ),
            # System Help & Greeting
            (
                IntentType.HELP,
                [
                    r"\bhelp\b",
                    r"\b(what can you do|options|commands|show commands|list commands)\b",
                ],
            ),
            (
                IntentType.GREETING,
                [
                    r"\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b",
                    r"\bhow are you\b",
                ],
            ),
            # Vehicle Sensors: Temperature Query
            (
                IntentType.GET_TEMPERATURE,
                [
                    r"\b(what is|check|tell me|get|read|current|how hot|how cold|how warm)\b.*?\btemperature\b",
                    r"\btemp in (the )?(car|cabin)\b",
                    r"\bhow (hot|cold|warm) is (it|the cabin|the car)\b",
                    r"\bcheck (the )?temperature\b",
                    r"^(cabin |indoor |inside |current )?temperature$",
                ],
            ),
            # Vehicle Sensors: Humidity Query
            (
                IntentType.GET_HUMIDITY,
                [
                    r"\b(what is|check|tell me|get|read|current|how humid)\b.*?\b(humidity|moisture)\b",
                    r"\bhow humid is (it|the cabin|the car)\b",
                    r"\bcheck (the )?(humidity|moisture)\b",
                    r"\bmoisture (level|sensor)\b",
                    r"^(cabin |indoor |inside |current )?humidity$",
                ],
            ),
            # Vehicle Sensors: Distance / Obstacle Query
            (
                IntentType.GET_DISTANCE,
                [
                    r"\b(how far|distance|range)\b.*?\b(obstacle|object|wall|barrier|front)\b",
                    r"\b(obstacle|object)\b.*?\b(distance|how far)\b",
                    r"\bhow close is (the )?(obstacle|object|wall)\b",
                    r"\bcheck (the )?(obstacle |distance )?sensor\b",
                    r"\bcheck (the )?distance\b",
                    r"\bproximity (reading|check|distance)\b",
                ],
            ),
            # Vehicle Sensors: Diagnostic Check
            (
                IntentType.GET_SENSOR_STATUS,
                [
                    r"\b(check|read|scan|status of|diagnose|status|report)\b.*?\bsensors?\b",
                    r"\bsensor (status|report|diagnostics|check|readings)\b",
                    r"\bare (the )?sensors working\b",
                    r"\b(report|give me)\b.*?\bsensors?\b",
                ],
            ),
            # Emergency Hazard Mode (Priority before general lighting)
            (
                IntentType.EMERGENCY_OFF,
                [
                    r"\b(disable|turn off|switch off|stop|deactivate)\b.*?\b(emergency|hazard)\b",
                    r"\b(hazard|emergency)\s*off\b",
                ],
            ),
            (
                IntentType.EMERGENCY_ON,
                [
                    r"\b(enable|turn on|switch on|start|activate)?\b.*?\b(emergency|hazard)\b",
                    r"\b(hazard|emergency)\s*on\b",
                    r"^(emergency|hazard)$",
                ],
            ),
            # Vehicle Lighting: Status Query
            (
                IntentType.GET_LIGHTS_STATUS,
                [
                    r"\b(are|is)\b.*?\b(lights|headlights|lamps|lighting)\b.*?\b(on|off|active)\b",
                    r"\b(check|status of|what is|get)\b.*?\b(lights|headlights|lamps|lighting)\b.*?(status)?",
                    r"\b(light|headlight|lighting) status\b",
                    r"\bcurrent light status\b",
                ],
            ),
            # Vehicle Lighting: Actuators
            (
                IntentType.LIGHTS_ON,
                [
                    r"^(?!.*(?:hazard|emergency)).*\b(turn|switch|power)\b.*?\b(lights|headlights|lamps|lighting)\b.*?\bon\b",
                    r"^(?!.*(?:hazard|emergency)).*\b(turn on|switch on|enable|activate|power on)\b.*?\b(lights|headlights|lamps|lighting)\b",
                    r"^(?!.*(?:hazard|emergency)).*\b(keep|leave)\b.*?\b(lights|headlights|lamps|lighting)\b.*?\b(on|running)\b",
                    r"^(lights|headlights) on$",
                ],
            ),
            (
                IntentType.LIGHTS_OFF,
                [
                    r"^(?!.*(?:hazard|emergency)).*\b(turn|switch|power)\b.*?\b(lights|headlights|lamps|lighting)\b.*?\boff\b",
                    r"^(?!.*(?:hazard|emergency)).*\b(turn off|switch off|disable|deactivate|power off|shut off|shut down)\b.*?\b(lights|headlights|lamps|lighting)\b",
                    r"^(?!.*(?:hazard|emergency)).*\b(shut)\b.*?\b(lights|headlights|lamps)\b.*?\b(off|down)\b",
                    r"^(?!.*(?:hazard|emergency)).*\b(leave|keep|stay)\b.*?\b(lights|headlights|lamps)\b.*?\boff\b",
                    r"^(?!.*(?:hazard|emergency)).*\b(lights|headlights|lamps)\b.*?\bstay\s+off\b",
                    r"^(lights|headlights) off$",
                ],
            ),
            # AC / Climate: Status Query
            (
                IntentType.GET_AC_STATUS,
                [
                    r"\bis (the )?(ac|air conditioning|climate control)\b.*?\b(on|running|off|active|turned on)\b",
                    r"\bcheck if (the )?(ac|air conditioning)\b.*?\b(on|running|off|active)\b",
                    r"\b(what is|check|get) (the )?(ac|air conditioning|climate) status\b",
                    r"\bac (state|status)\b",
                ],
            ),
            # AC / Climate: Actuators
            (
                IntentType.TURN_AC_ON,
                [
                    r"\b(turn|switch|power)\b.*?\b(ac|air conditioning|air conditioner|cooling|climate control)\b.*?\bon\b",
                    r"\b(turn on|switch on|enable|start|activate|power on)\b.*?\b(ac|air conditioning|air conditioner|cooling|climate control)\b",
                    r"\bkeep\b.*?\b(ac|air conditioning|cooling)\b.*?\b(on|running|active)\b",
                    r"\b(too hot|too warm|getting warm)\b.*?\b(ac|cooling)\b",
                    r"^ac on$",
                ],
            ),
            (
                IntentType.TURN_AC_OFF,
                [
                    r"\b(turn|switch|power)\b.*?\b(ac|air conditioning|air conditioner|cooling|climate control|cooling unit)\b.*?\boff\b",
                    r"\b(turn off|switch off|disable|stop|shut off|shut down|deactivate|power off)\b.*?\b(ac|air conditioning|air conditioner|cooling|climate control|cooling unit)\b",
                    r"\bleave\b.*?\b(ac|air conditioning)\b.*?\boff\b",
                    r"^ac off$",
                ],
            ),
            # Door Controls: Status Query
            (
                IntentType.GET_DOOR_STATUS,
                [
                    r"\b(check if|is|are)\b.*?\b(door|doors)\b.*?\b(open|closed|shut)\b",
                    r"\b(check|status of|what is) (the )?(door|doors)\b",
                    r"\bdoor status\b",
                ],
            ),
            # Door Controls: Open
            (
                IntentType.OPEN_DOOR,
                [
                    r"\b(open|opening)\b.*?\b(door|doors)\b",
                    r"\b(open|opening) (driver|passenger|both) door\b",
                ],
            ),
            # Door Controls: Close
            (
                IntentType.CLOSE_DOOR,
                [
                    r"\b(close|closing|shut|shutting)\b.*?\b(door|doors)\b",
                    r"\b(close|closing|shut|shutting) (driver|passenger|both) door\b",
                ],
            ),
        ]

        compiled_rules: List[Tuple[IntentType, List[Pattern]]] = []
        for intent, patterns in raw_rules:
            compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
            compiled_rules.append((intent, compiled_patterns))

        return compiled_rules

    @staticmethod
    def _extract_entities(intent: IntentType, clause: str) -> Dict[str, Any]:
        """Extracts domain-specific parameters (e.g. door targets)."""
        entities: Dict[str, Any] = {}
        if intent in (IntentType.OPEN_DOOR, IntentType.CLOSE_DOOR, IntentType.GET_DOOR_STATUS):
            if re.search(r"\b(driver|driver's|left)\b", clause):
                entities["door"] = "driver"
            elif re.search(r"\b(passenger|passenger's|right)\b", clause):
                entities["door"] = "passenger"
            elif re.search(r"\b(both|all)\b", clause) or ("driver" in clause and "passenger" in clause):
                entities["door"] = "both"
            elif intent in (IntentType.OPEN_DOOR, IntentType.CLOSE_DOOR):
                entities["door"] = "unspecified"
        return entities

    def _classify_atomic(self, clause: str) -> Optional[Tuple[IntentType, Dict[str, Any]]]:
        """Matches a single atomic clause against prioritized rules.

        Returns (IntentType, entities) or None if unmatched / unsupported.
        """
        # Guard against explicit unsupported feature attempts:
        # 1. Changing/setting temperature or thermostat
        if re.search(r"\b(set|change|adjust|increase|decrease|lower|raise)\b.*?\b(temperature|temp|ac|thermostat|degrees?)\b", clause):
            return None

        # 2. Locking / unlocking (lock hardware does not exist; only open/close servos)
        if re.search(r"\b(unlock|lock)\b", clause):
            return None

        # 3. Media, navigation, windows, seats, sunroof, mirror
        if re.search(r"\b(music|radio|song|spotify|volume|navigate|navigation|route|directions?|window|windows|sunroof|seat|mirror)\b", clause):
            return None

        for intent, patterns in self._rules:
            for pattern in patterns:
                if pattern.search(clause):
                    entities = self._extract_entities(intent, clause)
                    return intent, entities

        return None

    def _preprocess_utterance(self, text: str) -> Tuple[str, bool]:
        """Normalizes and handles polite prefixes/suffixes, commas, and negative contrast."""
        if not text:
            return "", False

        # Convert list commas to conjunctions before punctuation stripping
        converted = re.sub(r",\s*and\s+", " and ", text)
        converted = re.sub(r",\s*", " and ", converted)

        normalized = normalize_text(converted)
        if not normalized:
            return "", False

        cleaned = strip_polite_fillers(normalized)

        # Handle negative contrast: "turn off the ac, not the headlights"
        neg_match = re.search(
            r"^(.*?)\s+(?:not|do not)\s+(?:the\s+)?(lights|headlights|ac|air conditioning|driver door|passenger door|doors?)\s*$",
            cleaned,
        )
        if neg_match:
            affirmative_part = neg_match.group(1).strip()
            return affirmative_part, True

        return cleaned, False

    def _resolve_pronouns(self, clauses: List[str]) -> List[str]:
        """Resolves anaphoric pronouns ('turn them off', 'turn it off') from antecedent clauses."""
        resolved = []
        last_target = None

        for clause in clauses:
            curr = clause
            if re.search(r"\b(them|it)\b", curr):
                if last_target == "lights":
                    curr = re.sub(r"\b(them|it)\b", "the headlights", curr)
                elif last_target == "ac":
                    curr = re.sub(r"\b(them|it)\b", "the ac", curr)
                elif last_target == "doors":
                    curr = re.sub(r"\b(them|it)\b", "both doors", curr)

            resolved.append(curr)

            if re.search(r"\b(lights|headlights|lamps)\b", curr):
                last_target = "lights"
            elif re.search(r"\b(ac|air conditioning|air conditioner|cooling)\b", curr):
                last_target = "ac"
            elif re.search(r"\b(door|doors)\b", curr):
                last_target = "doors"

        return resolved

    def _segment_compound(self, text: str) -> List[str]:
        """Segments a compound utterance into ordered, atomic actionable clauses."""
        # 1. Handle sequential inversion: "before <clause_a>, <clause_b>"
        before_match = re.match(
            r"^before\s+(.*?)\s+(turn\s+on|turn\s+off|switch\s+on|switch\s+off|start|stop|activate|deactivate|open|close|shut)\s+(.*)$",
            text,
        )
        if before_match:
            clause_a = before_match.group(1).strip()
            verb_b = before_match.group(2).strip()
            rest_b = before_match.group(3).strip()
            clause_b = f"{verb_b} {rest_b}"
            return [clause_b, clause_a]

        # 2. Protect "driver and passenger doors" from being split on "and"
        text = re.sub(r"\b(driver and passenger|passenger and driver)\s+doors?\b", "both doors", text)

        # 3. Handle unpunctuated STT verb/target boundaries (e.g. from Whisper)
        targets_pat = r"(?:the\s+ac|ac|air\s+conditioning|the\s+headlights|headlights|the\s+lights|lights|driver\s+door|passenger\s+door|both\s+doors)"
        verbs_pat = r"(?:turn\s+on|turn\s+off|switch\s+on|switch\s+off|activate|deactivate|open|close|shut|start|stop)"
        text = re.sub(rf"({targets_pat})\s+({verbs_pat})", r"\1 and \2", text)
        text = re.sub(r"(headlights|lights)\s+(ac|air conditioning)", r"\1 and \2", text)

        # 4. Check for single-verb distributing over compound objects:
        # e.g. "turn off the ac and headlights", "turn on both ac and headlights"
        distrib_pattern = (
            r"^(turn\s+on|turn\s+off|switch\s+on|switch\s+off|power\s+on|power\s+off|activate|deactivate|enable|disable|start|stop)\s+"
            r"(?:both\s+)?(the\s+ac|the\s+air conditioning|ac|air conditioning|the\s+headlights|the\s+lights|headlights|lights)\s+"
            r"(?:and|as well as|plus)\s+"
            r"(?:the\s+ac|the\s+air conditioning|ac|air conditioning|the\s+headlights|the\s+lights|headlights|lights)$"
        )
        match_distrib = re.match(distrib_pattern, text)
        if match_distrib:
            verb = match_distrib.group(1).strip()
            parts = re.split(r"\s+(?:and|as well as|plus)\s+", text[len(verb):].strip())
            clauses = [f"{verb} {p.replace('both', '').strip()}" for p in parts]
            return clauses

        # 5. Split on conjunction boundaries
        cleaned_text = re.sub(r"\bfirst\b", "", text).strip()
        split_delims = r"\s+(?:and\s+then|then|after\s+that|but|and)\s+"
        raw_clauses = re.split(split_delims, cleaned_text)
        raw_clauses = [c.strip() for c in raw_clauses if c.strip()]

        # 6. Handle elliptical objects that inherit verbs:
        # e.g. ["turn on the headlights", "ac", "open driver door"]
        expanded_clauses = []
        last_verb = None
        for clause in raw_clauses:
            verb_match = re.match(r"^(turn\s+on|turn\s+off|switch\s+on|switch\s+off|activate|deactivate|open|close|shut|start|stop)\b", clause)
            if verb_match:
                last_verb = verb_match.group(1)
                expanded_clauses.append(clause)
            elif clause in ("ac", "the ac", "air conditioning", "the air conditioning", "headlights", "the headlights", "lights", "the lights") and last_verb:
                expanded_clauses.append(f"{last_verb} {clause}")
            else:
                expanded_clauses.append(clause)

        return self._resolve_pronouns(expanded_clauses)

    def classify(self, text: str) -> ParsedCommand:
        """Classifies an utterance into atomic or compound actions.

        Args:
            text: Raw speech-to-text transcript.

        Returns:
            ParsedCommand instance with ordered actions, status, and metadata.
        """
        normalized = normalize_text(text)
        if not normalized:
            return ParsedCommand(
                actions=[],
                status=CommandStatus.EMPTY,
                raw_text=text,
                normalized_text="",
                confidence=0.0,
                error_message="Empty speech transcript received.",
            )

        cleaned, was_negated = self._preprocess_utterance(text)
        if not cleaned:
            return ParsedCommand(
                actions=[],
                status=CommandStatus.EMPTY,
                raw_text=text,
                normalized_text=normalized,
                confidence=0.0,
            )

        # Single atomic check: if no compound conjunctions are present and not negated
        if not was_negated and not any(k in cleaned for k in [" and ", " then ", " but ", "before "]):
            atomic_res = self._classify_atomic(cleaned)
            if atomic_res is not None:
                intent, entities = atomic_res
                return ParsedCommand(
                    actions=[Action(type=intent, parameters=entities)],
                    status=CommandStatus.VALID,
                    raw_text=text,
                    normalized_text=normalized,
                    confidence=1.0,
                )

        # Compound sentence segmentation
        clauses = self._segment_compound(cleaned)

        if len(clauses) <= 1:
            target_clause = clauses[0] if clauses else cleaned
            res = self._classify_atomic(target_clause)
            if res is None:
                return ParsedCommand(
                    actions=[],
                    status=CommandStatus.UNSUPPORTED,
                    raw_text=text,
                    normalized_text=normalized,
                    confidence=0.0,
                    error_message=f"Unsupported command: '{target_clause}'",
                )
            intent, entities = res
            return ParsedCommand(
                actions=[Action(type=intent, parameters=entities)],
                status=CommandStatus.VALID,
                raw_text=text,
                normalized_text=normalized,
                confidence=1.0,
            )

        # Multi-clause evaluation (Compound command)
        parsed_actions: List[Action] = []
        for clause in clauses:
            res = self._classify_atomic(clause)
            if res is None:
                # SAFETY: Reject entire compound command if any sub-action is unsupported
                return ParsedCommand(
                    actions=[],
                    status=CommandStatus.REJECTED,
                    raw_text=text,
                    normalized_text=normalized,
                    confidence=0.0,
                    error_message=f"Compound command rejected: unsupported sub-action '{clause}'.",
                )
            intent, entities = res
            # SAFETY: In compound commands, ambiguous door target cannot be defaulted
            if intent in (IntentType.OPEN_DOOR, IntentType.CLOSE_DOOR) and entities.get("door") == "unspecified":
                return ParsedCommand(
                    actions=[],
                    status=CommandStatus.REJECTED,
                    raw_text=text,
                    normalized_text=normalized,
                    confidence=0.0,
                    error_message="Compound command rejected: ambiguous door target (please specify driver, passenger, or both).",
                )
            parsed_actions.append(Action(type=intent, parameters=entities))

        return ParsedCommand(
            actions=parsed_actions,
            status=CommandStatus.VALID,
            raw_text=text,
            normalized_text=normalized,
            confidence=1.0,
        )
