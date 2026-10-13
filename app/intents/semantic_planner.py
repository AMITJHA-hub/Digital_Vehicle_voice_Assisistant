"""Deterministic Semantic / Action-Planning Layer for In-Vehicle Voice Assistant.

Provides structured pre-dispatch action planning, entity resolution, collective
and contextual references, pronoun anaphora resolution, negation and exclusions,
conversation context tracking, and safety validation.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Set, Tuple, Any

from .models import IntentType, CommandStatus, Action, ActionPlan, ParsedCommand
from .normalizer import normalize_text, strip_polite_fillers

logger = logging.getLogger(__name__)


class ContextTracker:
    """Explicit, lightweight conversation context tracker for elliptical references."""

    def __init__(self, ttl_seconds: float = 60.0):
        self.ttl_seconds = ttl_seconds
        self.last_command: Optional[ParsedCommand] = None
        self.last_actions: List[Action] = []
        self.last_targets: List[str] = []
        self.last_verb: Optional[str] = None
        self.timestamp: float = 0.0

    def update(self, command: ParsedCommand) -> None:
        """Updates conversation context from a successfully validated command."""
        if command.status != CommandStatus.VALID or not command.actions:
            return

        self.last_command = command
        self.last_actions = list(command.actions)
        self.timestamp = time.time()

        targets: List[str] = []
        verbs: List[str] = []
        for act in command.actions:
            name = act.type.name
            if "LIGHTS" in name:
                targets.append("lights")
                verbs.append("on" if "ON" in name else "off")
            elif "AC" in name:
                targets.append("ac")
                verbs.append("on" if "ON" in name else "off")
            elif "DOOR" in name:
                door_param = act.parameters.get("door", "both")
                targets.append(f"{door_param}_door" if "door" not in door_param else door_param)
                verbs.append("open" if "OPEN" in name else "close")
            elif "EMERGENCY" in name:
                targets.append("emergency")
                verbs.append("on" if "ON" in name else "off")
            elif act.type == IntentType.GLOBAL_OFF:
                targets.extend(["lights", "ac", "emergency"])
                verbs.append("off")
            elif act.type == IntentType.GLOBAL_ON:
                targets.extend(["lights", "ac"])
                verbs.append("on")

        # Deduplicate while preserving order
        seen = set()
        dedup_targets = []
        for t in targets:
            if t not in seen:
                seen.add(t)
                dedup_targets.append(t)

        self.last_targets = dedup_targets
        self.last_verb = verbs[-1] if verbs else None

    def is_valid(self, now: Optional[float] = None) -> bool:
        """Returns True if context contains actionable targets and has not expired."""
        if not self.last_actions or not self.last_targets:
            return False
        curr_time = now if now is not None else time.time()
        return (curr_time - self.timestamp) <= self.ttl_seconds

    def reset(self) -> None:
        """Clears all conversation history."""
        self.last_command = None
        self.last_actions = []
        self.last_targets = []
        self.last_verb = None
        self.timestamp = 0.0


class SemanticPlanner:
    """Deterministic semantic parser and action planner."""

    # Controllable allowlists for safe global operations
    GLOBAL_OFF_ALLOWLIST: List[IntentType] = [
        IntentType.LIGHTS_OFF,
        IntentType.TURN_AC_OFF,
        IntentType.EMERGENCY_OFF,
    ]

    GLOBAL_ON_ALLOWLIST: List[IntentType] = [
        IntentType.LIGHTS_ON,
        IntentType.TURN_AC_ON,
    ]

    def __init__(self, context_tracker: Optional[ContextTracker] = None):
        self.context = context_tracker or ContextTracker()

    def plan(
        self,
        raw_text: str,
        context: Optional[ContextTracker] = None,
    ) -> Optional[ActionPlan]:
        """Inspects text for natural-language patterns, exclusions, pronouns, or global commands.

        Returns:
            ActionPlan if a deterministic semantic rule or structure resolved the utterance,
            or None if atomic SVM / syntax classifier should handle standard single commands.
        """
        active_context = context or self.context

        normalized = normalize_text(raw_text)
        if not normalized:
            return None

        cleaned = strip_polite_fillers(normalized)

        # ----------------------------------------------------------------------
        # 1. CANCELLATION / ABORT
        # ----------------------------------------------------------------------
        if re.search(r"^(cancel|cancel that|abort|never mind|stop that)$", cleaned):
            return ActionPlan(
                actions=[],
                status=CommandStatus.VALID,
                clarification_prompt="Command cancelled. No vehicle action taken.",
                validation_notes=["User explicitly cancelled operation."],
            )

        # ----------------------------------------------------------------------
        # 2. CONTRADICTION DETECTION
        # ----------------------------------------------------------------------
        # e.g., "turn off the ac but keep it running", "turn the ac off but keep it running"
        if re.search(r"\b(?:turn|switch|shut)\b.*?\b(?:ac|air\s+conditioning)\b.*?\b(?:off|down)\b.*?\b(?:keep\s+it\s+running|keep\s+running|leave\s+on|stay\s+on)\b", cleaned) or \
           re.search(r"\b(?:turn\s+off|switch\s+off|shut\s+down)\b.*?\b(?:ac|air\s+conditioning)\b.*?\b(?:keep\s+it\s+running|keep\s+running|leave\s+on|stay\s+on)\b", cleaned) or \
           re.search(r"\b(?:keep\s+it\s+running|keep\s+running)\b.*?\b(?:turn|switch|shut)\b.*?\b(?:ac|air\s+conditioning)\b.*?\b(?:off|down)\b", cleaned):
            return ActionPlan(
                actions=[],
                status=CommandStatus.REJECTED,
                is_contradictory=True,
                clarification_prompt="Contradictory command: cannot turn off the air conditioning while keeping it running. Please clarify.",
                validation_notes=["Detected conflicting 'turn off' and 'keep running' for AC."],
            )

        if re.search(r"\b(?:turn|switch)\b.*?\b(?:lights|headlights)\b.*?\b(?:on)\b.*?\b(?:leave\s+them\s+off|keep\s+them\s+off|stay\s+off)\b", cleaned) or \
           re.search(r"\b(?:turn|switch)\b.*?\b(?:lights|headlights)\b.*?\b(?:off)\b.*?\b(?:leave\s+them\s+on|keep\s+them\s+on|stay\s+on)\b", cleaned):
            return ActionPlan(
                actions=[],
                status=CommandStatus.REJECTED,
                is_contradictory=True,
                clarification_prompt="Contradictory command: conflicting on/off instructions given for vehicle lights. Please clarify.",
                validation_notes=["Detected simultaneous on/off instructions for lights."],
            )

        # ----------------------------------------------------------------------
        # 3. CONTEXTUAL INQUIRIES & INDIRECT REFERENCES (Clarifications)
        # ----------------------------------------------------------------------
        # A. Headlights clarification ("too dark") - only if not combined with an explicit action verb
        if (re.search(r"\b(?:too\s+dark|cabin\s+is\s+too\s+dark|can(?:not|\'t)\s+see)\b", cleaned)) and \
           not re.search(r"\b(?:turn|switch|start|power|activate)\b", cleaned):
            return ActionPlan(
                actions=[Action(type=IntentType.CLARIFY, parameters={"target": "headlights"})],
                status=CommandStatus.VALID,
                clarification_prompt="It appears dark. Would you like me to turn on the headlights?",
                validation_notes=["Contextual lighting discomfort; prompt user rather than auto-actuating."],
            )

        # B. AC clarification ("getting hot", "too warm") - only if not combined with an explicit action verb
        if re.search(r"\b(?:getting\s+(?:really\s+)?hot|really\s+hot|too\s+hot|too\s+warm|cabin\s+(?:is|feels)\s+(?:too\s+|really\s+)?(?:hot|warm)|car\s+is\s+(?:too\s+|getting\s+)?warm)\b", cleaned) and \
           not re.search(r"\b(?:turn|switch|start|power|activate)\b", cleaned):
            return ActionPlan(
                actions=[Action(type=IntentType.CLARIFY, parameters={"target": "ac"})],
                status=CommandStatus.VALID,
                clarification_prompt="It seems warm in the cabin. Would you like me to turn on the air conditioning?",
                validation_notes=["Contextual temperature discomfort; prompt user rather than auto-actuating."],
            )

        # C. Humidity sensor inquiry ("too humid")
        if re.search(r"\b(too humid|cabin is humid|feels humid|moisture is high)\b", cleaned):
            return ActionPlan(
                actions=[Action(type=IntentType.CLARIFY, parameters={"target": "humidity"})],
                status=CommandStatus.VALID,
                clarification_prompt="Cabin humidity can be checked with the sensor. Would you like me to read the humidity sensor?",
                validation_notes=["Humidity query offered; no unsupported dehumidifier invented."],
            )

        # D. Comfort ambiguity ("make the cabin more comfortable")
        if re.search(r"\bmake\s+(?:the\s+)?(?:cabin|car)\s+(?:more\s+)?comfortable\b", cleaned):
            return ActionPlan(
                actions=[Action(type=IntentType.CLARIFY, parameters={"target": "comfort"})],
                status=CommandStatus.VALID,
                clarification_prompt="I can turn on the air conditioning or check the cabin temperature and humidity. What would you like to do?",
                validation_notes=["Ambiguous comfort request clarified across available capabilities."],
            )

        # ----------------------------------------------------------------------
        # 4. PURE NEGATIVE PROHIBITIONS (No affirmative action requested)
        # ----------------------------------------------------------------------
        if re.search(r"^(?:don\'t|do not)\s+touch\s+(?:the\s+)?(?:doors?|locks?)\s*$", cleaned) or \
           re.search(r"^leave\s+(?:the\s+)?(?:doors?)\s+alone\s*$", cleaned):
            return ActionPlan(
                actions=[],
                status=CommandStatus.VALID,
                excluded_targets=["doors"],
                clarification_prompt="Understood, leaving the doors unchanged.",
                validation_notes=["Prohibition instruction confirmed without actuation."],
            )

        if re.search(r"^(?:don\'t|do not)\s+open\s+(?:the\s+)?passenger\s+door\s*$", cleaned):
            return ActionPlan(
                actions=[],
                status=CommandStatus.VALID,
                excluded_targets=["passenger_door"],
                clarification_prompt="Understood, leaving the passenger door unchanged.",
                validation_notes=["Single door prohibition confirmed without actuation."],
            )

        # ----------------------------------------------------------------------
        # 5. NEGATION & EXCLUSIONS WITH GLOBAL COMMANDS
        # ----------------------------------------------------------------------
        # "turn off everything except the headlights" / "switch everything off except lights"
        match_global_excl = re.search(
            r"^(?:turn\s+off|switch\s+off|shut\s+down)\s+(?:everything|all\s+systems)\s+except\s+(?:the\s+)?(headlights|lights|ac|air\s+conditioning)\b",
            cleaned,
        )
        if match_global_excl:
            target_excluded = match_global_excl.group(1).lower()
            plan_actions = []
            excluded = []
            if "light" in target_excluded:
                excluded.append("headlights")
                plan_actions = [Action(IntentType.TURN_AC_OFF), Action(IntentType.EMERGENCY_OFF)]
            elif "ac" in target_excluded:
                excluded.append("ac")
                plan_actions = [Action(IntentType.LIGHTS_OFF), Action(IntentType.EMERGENCY_OFF)]

            return ActionPlan(
                actions=plan_actions,
                excluded_targets=excluded,
                status=CommandStatus.VALID,
                validation_notes=[f"Global OFF executed excluding {excluded}."],
            )

        # "leave the lights on and switch everything else off"
        if re.search(r"\bleave\s+(?:the\s+)?(?:lights|headlights)\s+on\s+and\s+(?:switch|turn)\s+everything\s+else\s+off\b", cleaned) or \
           re.search(r"\bswitch\s+everything\s+else\s+off\s+and\s+leave\s+(?:the\s+)?(?:lights|headlights)\s+on\b", cleaned):
            return ActionPlan(
                actions=[Action(IntentType.TURN_AC_OFF), Action(IntentType.EMERGENCY_OFF)],
                excluded_targets=["headlights"],
                status=CommandStatus.VALID,
                validation_notes=["Lights preserved ON; everything else turned off."],
            )

        # ----------------------------------------------------------------------
        # 6. NEGATION / EXCLUSIONS ON COMPONENT COMMANDS
        # ----------------------------------------------------------------------
        # "don't touch the doors; turn the lights off" / "turn the lights off and don't touch the doors"
        if re.search(r"(?:do\s+not|dont|don\'t)\s+touch\s+(?:the\s+)?doors?\b.*?\b(?:turn|switch)\b.*?\b(?:lights|headlights)\b.*?\boff\b", cleaned) or \
           re.search(r"\b(?:turn|switch)\b.*?\b(?:lights|headlights)\b.*?\boff\b.*?\b(?:do\s+not|dont|don\'t)\s+touch\s+(?:the\s+)?doors?\b", cleaned):
            return ActionPlan(
                actions=[Action(IntentType.LIGHTS_OFF)],
                excluded_targets=["doors"],
                status=CommandStatus.VALID,
                validation_notes=["Doors excluded; lights turned off."],
            )

        # "turn off the ac, not the headlights" / "turn on the ac but leave lights alone"
        match_contrast = re.search(
            r"^(turn\s+on|turn\s+off|switch\s+on|switch\s+off)\s+(?:the\s+)?(ac|air\s+conditioning|lights|headlights)\s+(?:not|but\s+not)\s+(?:the\s+)?(lights|headlights|ac|air\s+conditioning)\b",
            cleaned,
        )
        if match_contrast:
            verb = match_contrast.group(1)
            target = match_contrast.group(2)
            neg_target = match_contrast.group(3)
            is_on = "on" in verb
            action_type = None
            if "ac" in target:
                action_type = IntentType.TURN_AC_ON if is_on else IntentType.TURN_AC_OFF
            else:
                action_type = IntentType.LIGHTS_ON if is_on else IntentType.LIGHTS_OFF

            return ActionPlan(
                actions=[Action(action_type)],
                excluded_targets=[neg_target],
                status=CommandStatus.VALID,
                validation_notes=[f"{target} actuated; {neg_target} explicitly excluded."],
            )

        if re.search(r"\bturn\s+(?:the\s+)?ac\s+on\s*,\s*but\s+leave\s+(?:the\s+)?lights\s+alone\b", cleaned):
            return ActionPlan(
                actions=[Action(IntentType.TURN_AC_ON)],
                excluded_targets=["lights"],
                status=CommandStatus.VALID,
                validation_notes=["AC turned on; lights left untouched."],
            )

        # "close the driver door but leave the passenger door open"
        if re.search(r"\bclose\s+(?:the\s+)?driver\s+door\s+(?:but|and)\s+leave\s+(?:the\s+)?passenger\s+door\s+open\b", cleaned):
            return ActionPlan(
                actions=[Action(IntentType.CLOSE_DOOR, {"door": "driver"})],
                excluded_targets=["passenger_door"],
                status=CommandStatus.VALID,
                validation_notes=["Driver door closed; passenger door confirmed open/untouched."],
            )

        # ----------------------------------------------------------------------
        # 7. CONTEXT-DEPENDENT & ELLIPTICAL REFERENCES
        # ----------------------------------------------------------------------
        # "switch both of those off" / "turn all of them off" / "disable both systems" / "and switch those off"
        if re.search(r"^(?:switch|turn|power)\s+(?:both\s+of\s+those|all\s+of\s+them|both\s+systems|those|them)\s+off\b", cleaned) or \
           re.search(r"^(?:disable|deactivate)\s+both\s+systems\b", cleaned) or \
           re.search(r"^and\s+(?:switch|turn)\s+those\s+off\b", cleaned):
            if active_context and active_context.is_valid():
                plan_actions = []
                for tgt in active_context.last_targets:
                    if tgt == "lights":
                        plan_actions.append(Action(IntentType.LIGHTS_OFF))
                    elif tgt == "ac":
                        plan_actions.append(Action(IntentType.TURN_AC_OFF))
                    elif tgt in ("driver_door", "passenger_door", "both_doors"):
                        door_p = tgt.replace("_door", "")
                        plan_actions.append(Action(IntentType.CLOSE_DOOR, {"door": door_p}))
                    elif tgt == "emergency":
                        plan_actions.append(Action(IntentType.EMERGENCY_OFF))

                if plan_actions:
                    return ActionPlan(
                        actions=plan_actions,
                        status=CommandStatus.VALID,
                        validation_notes=[f"Resolved context targets {active_context.last_targets} to OFF."],
                    )

            # No context or expired context
            return ActionPlan(
                actions=[],
                status=CommandStatus.AMBIGUOUS,
                clarification_prompt="Cannot determine which systems to switch off. Please specify headlights, AC, or doors.",
                validation_notes=["Elliptical reference without active conversation antecedent."],
            )

        # "both on" / "switch the two systems back on"
        if re.search(r"^(?:both\s+on|switch\s+(?:the\s+)?two\s+systems\s+(?:back\s+)?on)$", cleaned):
            if active_context and active_context.is_valid():
                plan_actions = []
                for tgt in active_context.last_targets:
                    if tgt == "lights":
                        plan_actions.append(Action(IntentType.LIGHTS_ON))
                    elif tgt == "ac":
                        plan_actions.append(Action(IntentType.TURN_AC_ON))
                    elif tgt in ("driver_door", "passenger_door", "both_doors"):
                        door_p = tgt.replace("_door", "")
                        plan_actions.append(Action(IntentType.OPEN_DOOR, {"door": door_p}))

                if plan_actions:
                    return ActionPlan(
                        actions=plan_actions,
                        status=CommandStatus.VALID,
                        validation_notes=[f"Resolved context targets {active_context.last_targets} to ON."],
                    )

            return ActionPlan(
                actions=[],
                status=CommandStatus.AMBIGUOUS,
                clarification_prompt="Cannot determine which systems to turn on. Please specify headlights, AC, or doors.",
                validation_notes=["Elliptical 'both on' without active antecedent."],
            )

        # "the headlights too" / "do the same for the lights"
        if re.search(r"^(?:the\s+)?(?:headlights|lights)\s+too\b|^do\s+the\s+same\s+for\s+(?:the\s+)?(?:lights|headlights)\b", cleaned):
            if active_context and active_context.is_valid() and active_context.last_verb:
                verb = active_context.last_verb
                act = IntentType.LIGHTS_ON if verb in ("on", "open") else IntentType.LIGHTS_OFF
                return ActionPlan(
                    actions=[Action(act)],
                    status=CommandStatus.VALID,
                    validation_notes=[f"Applied antecedent verb '{verb}' to headlights."],
                )
            return ActionPlan(
                actions=[],
                status=CommandStatus.AMBIGUOUS,
                clarification_prompt="I don't know whether to turn headlights on or off. Please specify.",
                validation_notes=["Elliptical headlights reference missing prior action verb."],
            )

        # "same for the passenger side" / "now the other one"
        if re.search(r"^(?:same\s+for\s+(?:the\s+)?passenger\s+side|now\s+the\s+other\s+one)$", cleaned):
            if active_context and active_context.is_valid() and active_context.last_verb:
                verb = active_context.last_verb
                act = IntentType.OPEN_DOOR if verb == "open" else IntentType.CLOSE_DOOR
                return ActionPlan(
                    actions=[Action(act, {"door": "passenger"})],
                    status=CommandStatus.VALID,
                    validation_notes=[f"Applied antecedent door verb '{verb}' to passenger door."],
                )
            return ActionPlan(
                actions=[],
                status=CommandStatus.AMBIGUOUS,
                clarification_prompt="Please specify which door action you would like to perform.",
                validation_notes=["Elliptical door command missing antecedent."],
            )

        # "i said turn them off not on"
        if re.search(r"^i\s+said\s+turn\s+them\s+off\s+(?:not|and\s+not)\s+on\b", cleaned):
            if active_context and active_context.is_valid():
                plan_actions = []
                for tgt in active_context.last_targets:
                    if tgt == "lights":
                        plan_actions.append(Action(IntentType.LIGHTS_OFF))
                    elif tgt == "ac":
                        plan_actions.append(Action(IntentType.TURN_AC_OFF))
                if plan_actions:
                    return ActionPlan(
                        actions=plan_actions,
                        status=CommandStatus.VALID,
                        validation_notes=["User correction inverted targets to OFF."],
                    )
            return ActionPlan(
                actions=[],
                status=CommandStatus.AMBIGUOUS,
                clarification_prompt="Which systems would you like to turn off?",
                validation_notes=["Correction without clear antecedent."],
            )

        # ----------------------------------------------------------------------
        # 8. GLOBAL COMMANDS (Allowlist & Safe Policy)
        # ----------------------------------------------------------------------
        # Global OFF: "turn off everything", "switch everything off", "shut everything down", "power down all systems"
        global_off_patterns = [
            r"^(?:turn|switch|power)\s+(?:everything|it\s+all|all\s+the\s+systems|all\s+the\s+vehicle\s+systems|all\s+systems|all\s+the\s+electronics|the\s+electricals|the\s+nonessential\s+stuff)\s+off$",
            r"^(?:turn\s+off|switch\s+off|shut\s+down|power\s+down|deactivate)\s+(?:everything|it\s+all|all\s+the\s+systems|all\s+the\s+vehicle\s+systems|all\s+systems|all\s+the\s+electronics|all\s+controllable\s+systems|all\s+cabin\s+systems|the\s+electricals|cabin\s+systems|the\s+nonessential\s+stuff)$",
            r"^(?:turn\s+off|switch\s+off)\s+all\s+(?:the\s+)?lights\s+and\s+climate(?:\s+controls)?$",
            r"^everything\s+off$",
            r"^i\s+want\s+everything\s+switched\s+off$",
            r"^can\s+you\s+shut\s+everything\s+down$",
        ]
        for pat in global_off_patterns:
            if re.search(pat, cleaned):
                return ActionPlan(
                    actions=[Action(IntentType.GLOBAL_OFF)],
                    status=CommandStatus.VALID,
                    validation_notes=["Global OFF: Safe allowlist (lights, AC, emergency). Doors untouched."],
                )

        # Global ON: "turn everything on", "enable everything", "restore all systems"
        global_on_patterns = [
            r"^(?:turn|switch|power)\s+(?:everything|all\s+systems|all\s+the\s+systems)\s+on$",
            r"^(?:turn\s+on|switch\s+on|enable|activate|restore)\s+(?:everything|all\s+systems|all\s+the\s+systems|all\s+the\s+systems\s+again|all\s+controllable\s+systems)$",
            r"^turn\s+everything\s+back\s+on$",
            r"^everything\s+on$",
        ]
        for pat in global_on_patterns:
            if re.search(pat, cleaned):
                return ActionPlan(
                    actions=[Action(IntentType.GLOBAL_ON)],
                    status=CommandStatus.VALID,
                    validation_notes=["Global ON: Safe primary accessories (lights, AC). Doors kept safe."],
                )

        # ----------------------------------------------------------------------
        # 9. PRONOUN ANAPHORA & COMPOUND SEQUENCING IN SAME UTTERANCE
        # ----------------------------------------------------------------------
        # "turn on the lights and leave them on" / "switch on lights keep those running"
        if re.search(r"^(?:turn\s+on|switch\s+on)\s+(?:the\s+)?(?:lights|headlights)\s+and\s+(?:leave|keep)\s+(?:them|those)\s+(?:on|running)$", cleaned):
            return ActionPlan(
                actions=[Action(IntentType.LIGHTS_ON)],
                status=CommandStatus.VALID,
                validation_notes=["Headlights turned on; state preservation confirmed."],
            )

        # "turn the ac on and leave everything else alone"
        if re.search(r"^turn\s+(?:the\s+)?ac\s+on\s+and\s+leave\s+everything\s+else\s+alone$", cleaned):
            return ActionPlan(
                actions=[Action(IntentType.TURN_AC_ON)],
                status=CommandStatus.VALID,
                validation_notes=["AC turned on; other components isolated."],
            )

        # "switch on the ac and turn it off again" / "turn on the ac and then turn it off"
        if re.search(r"^(?:turn\s+on|switch\s+on)\s+(?:the\s+)?ac\s+(?:and\s+then|and)\s+(?:turn\s+it\s+off|switch\s+it\s+off|turn\s+off)(?:\s+again)?$", cleaned):
            return ActionPlan(
                actions=[Action(IntentType.TURN_AC_ON), Action(IntentType.TURN_AC_OFF)],
                status=CommandStatus.VALID,
                validation_notes=["Sequential ordered actions: AC ON, then AC OFF."],
            )

        # "open the driver door and then close it" / "open the driver door and close it"
        if re.search(r"^open\s+(?:the\s+)?driver\s+door\s+(?:and\s+then|and)\s+(?:close\s+it|shut\s+it)$", cleaned):
            return ActionPlan(
                actions=[
                    Action(IntentType.OPEN_DOOR, {"door": "driver"}),
                    Action(IntentType.CLOSE_DOOR, {"door": "driver"}),
                ],
                status=CommandStatus.VALID,
                validation_notes=["Resolved pronoun 'it' to driver door: OPEN then CLOSE."],
            )

        # "turn on the headlights and ac then turn those off"
        if re.search(r"^turn\s+on\s+(?:the\s+)?(?:headlights|lights)\s+and\s+ac\s+(?:and\s+then|then)\s+turn\s+those\s+off$", cleaned):
            return ActionPlan(
                actions=[Action(IntentType.LIGHTS_OFF), Action(IntentType.TURN_AC_OFF)],
                status=CommandStatus.VALID,
                validation_notes=["Pronoun 'those' refers to headlights and AC; inverted to OFF."],
            )

        # "turn on the lights. actually, switch them off" / "no, switch the headlights off instead"
        if re.search(r"actually\s*,?\s*(?:switch|turn)\s+(?:them|the\s+lights|the\s+headlights)\s+off", cleaned) or \
           re.search(r"^no\s*,?\s*(?:switch|turn)\s+(?:the\s+)?(?:headlights|lights)\s+off\s+instead$", cleaned):
            return ActionPlan(
                actions=[Action(IntentType.LIGHTS_OFF)],
                status=CommandStatus.VALID,
                validation_notes=["User correction superseded affirmative instruction."],
            )

        # "turn the lights and ac off" / "lights and ac off"
        if re.search(r"^(?:turn\s+(?:the\s+)?lights\s+and\s+ac\s+off|lights\s+and\s+ac\s+off)$", cleaned):
            return ActionPlan(
                actions=[Action(IntentType.LIGHTS_OFF), Action(IntentType.TURN_AC_OFF)],
                status=CommandStatus.VALID,
                validation_notes=["Collective object: lights and AC turned OFF."],
            )

        # If none of the specialized semantic structures matched, return None so
        # the standard SVM / compound clause classifier handles it.
        return None
