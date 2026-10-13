"""Data models and enum definitions for intent and action parsing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IntentType(str, Enum):
    """Enumeration of all supported vehicle voice assistant intents and actions."""
    GREETING = "GREETING"
    GET_TEMPERATURE = "GET_TEMPERATURE"
    GET_HUMIDITY = "GET_HUMIDITY"
    GET_DISTANCE = "GET_DISTANCE"
    GET_SENSOR_STATUS = "GET_SENSOR_STATUS"
    OPEN_DOOR = "OPEN_DOOR"
    CLOSE_DOOR = "CLOSE_DOOR"
    GET_DOOR_STATUS = "GET_DOOR_STATUS"
    TURN_AC_ON = "TURN_AC_ON"
    TURN_AC_OFF = "TURN_AC_OFF"
    GET_AC_STATUS = "GET_AC_STATUS"
    LIGHTS_ON = "LIGHTS_ON"
    LIGHTS_OFF = "LIGHTS_OFF"
    GET_LIGHTS_STATUS = "GET_LIGHTS_STATUS"
    EMERGENCY_ON = "EMERGENCY_ON"
    EMERGENCY_OFF = "EMERGENCY_OFF"
    STOP_LISTENING = "STOP_LISTENING"
    HELP = "HELP"
    EXIT = "EXIT"
    COMPOUND = "COMPOUND"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"


# Alias for explicit action modeling
ActionType = IntentType


class CommandStatus(str, Enum):
    """Validation and execution status of a parsed command."""
    VALID = "VALID"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"
    REJECTED = "REJECTED"
    EMPTY = "EMPTY"


@dataclass
class Action:
    """Represents a single atomic vehicle action or query with typed parameters."""
    type: IntentType
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value if hasattr(self.type, "value") else str(self.type),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        type_str = data.get("type", "UNSUPPORTED")
        intent = getattr(IntentType, type_str, IntentType.UNSUPPORTED)
        return cls(type=intent, parameters=dict(data.get("parameters", {})))


@dataclass
class ParsedCommand:
    """Represents a categorized user request, containing an ordered list of actions.

    Maintains full backward compatibility with single-intent code.
    """
    actions: List[Action] = field(default_factory=list)
    status: CommandStatus = CommandStatus.VALID
    raw_text: str = ""
    normalized_text: str = ""
    confidence: float = 1.0
    error_message: Optional[str] = None

    def __init__(
        self,
        actions: Optional[List[Action]] = None,
        status: CommandStatus = CommandStatus.VALID,
        raw_text: str = "",
        normalized_text: str = "",
        confidence: float = 1.0,
        error_message: Optional[str] = None,
        intent: Optional[IntentType] = None,
        entities: Optional[Dict[str, Any]] = None,
    ):
        self.raw_text = raw_text
        self.normalized_text = normalized_text
        self.confidence = confidence
        self.error_message = error_message
        self.status = status

        if actions is not None:
            self.actions = list(actions)
        elif intent is not None:
            if intent == IntentType.UNSUPPORTED:
                self.status = CommandStatus.UNSUPPORTED
                self.actions = []
            elif intent == IntentType.AMBIGUOUS:
                self.status = CommandStatus.AMBIGUOUS
                self.actions = []
            else:
                self.actions = [Action(type=intent, parameters=entities or {})]
        else:
            self.actions = []

    @property
    def intent(self) -> IntentType:
        """Backward compatibility: returns primary intent, COMPOUND, AMBIGUOUS, or UNSUPPORTED."""
        if self.status == CommandStatus.AMBIGUOUS:
            return IntentType.AMBIGUOUS
        if self.status != CommandStatus.VALID:
            return IntentType.UNSUPPORTED
        if not self.actions:
            return IntentType.UNSUPPORTED
        if len(self.actions) == 1:
            return self.actions[0].type
        return IntentType.COMPOUND

    @property
    def entities(self) -> Dict[str, Any]:
        """Backward compatibility: returns parameters of the first action."""
        if self.actions:
            return dict(self.actions[0].parameters)
        return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "actions": [a.to_dict() for a in self.actions],
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "error_message": self.error_message,
        }
