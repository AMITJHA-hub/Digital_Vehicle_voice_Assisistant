"""Command dispatcher and spoken response generator."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from ..config import config, AppConfig
from ..hardware.interface import AbstractVehicleHardware
from ..intents.models import IntentType, CommandStatus, ParsedCommand


@dataclass
class DispatchResult:
    """Outcome of an orchestrated voice command."""
    intent: IntentType
    response_text: str
    success: bool
    should_exit: bool = False
    should_stop_listening: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommandDispatcher:
    """Maps parsed intents to vehicle hardware actions and spoken responses."""

    def __init__(
        self,
        hardware: AbstractVehicleHardware,
        app_config: Optional[AppConfig] = None,
    ):
        self.hardware = hardware
        self.config = app_config or config

    def dispatch(self, command: ParsedCommand) -> DispatchResult:
        """Executes the corresponding vehicle action(s) for the parsed intent."""
        if command.status == CommandStatus.EMPTY:
            return DispatchResult(
                intent=command.intent,
                response_text="No command was heard.",
                success=False,
            )

        if command.status == CommandStatus.AMBIGUOUS:
            return self._handle_ambiguous(command)

        if command.status == CommandStatus.REJECTED:
            msg = (command.plan and command.plan.clarification_prompt) or command.error_message or "I cannot execute this compound command because it contains unsupported or ambiguous actions."
            return DispatchResult(
                intent=IntentType.UNSUPPORTED,
                response_text=msg,
                success=False,
            )

        if command.status == CommandStatus.VALID and not command.actions and command.plan and command.plan.clarification_prompt:
            return DispatchResult(
                intent=command.intent,
                response_text=command.plan.clarification_prompt,
                success=True,
            )

        if command.status == CommandStatus.UNSUPPORTED or not command.actions:
            return self._handle_unsupported(command)

        if len(command.actions) == 1:
            handler_name = f"_handle_{command.actions[0].type.name.lower()}"
            handler = getattr(self, handler_name, self._handle_unsupported)
            return handler(command)

        return self._handle_compound(command)

    def _handle_greeting(self, cmd: ParsedCommand) -> DispatchResult:
        return DispatchResult(
            intent=cmd.intent,
            response_text="Hello! How can I assist you with your vehicle today?",
            success=True,
        )

    def _handle_get_temperature(self, cmd: ParsedCommand) -> DispatchResult:
        reading = self.hardware.get_temperature()
        prefix = "[Simulated] " if reading.is_simulated else ""
        if reading.status != "OK":
            text = f"{prefix}Unable to read temperature sensor. Status: {reading.status}."
            return DispatchResult(intent=cmd.intent, response_text=text, success=False)

        text = f"{prefix}Cabin temperature is {reading.value:.1f} degrees Celsius."
        return DispatchResult(
            intent=cmd.intent,
            response_text=text,
            success=True,
            metadata={"reading": reading},
        )

    def _handle_get_humidity(self, cmd: ParsedCommand) -> DispatchResult:
        reading = self.hardware.get_humidity()
        prefix = "[Simulated] " if reading.is_simulated else ""
        if reading.status != "OK":
            text = f"{prefix}Unable to read humidity sensor. Status: {reading.status}."
            return DispatchResult(intent=cmd.intent, response_text=text, success=False)

        text = f"{prefix}Cabin relative humidity is {reading.value:.0f} percent."
        return DispatchResult(
            intent=cmd.intent,
            response_text=text,
            success=True,
            metadata={"reading": reading},
        )

    def _handle_get_distance(self, cmd: ParsedCommand) -> DispatchResult:
        reading = self.hardware.get_distance()
        prefix = "[Simulated] " if reading.is_simulated else ""
        if reading.status != "OK":
            text = f"{prefix}Obstacle sensor unreachable. Status: {reading.status}."
            return DispatchResult(intent=cmd.intent, response_text=text, success=False)

        text = f"{prefix}Distance to obstacle is {reading.value:.1f} centimeters."
        return DispatchResult(
            intent=cmd.intent,
            response_text=text,
            success=True,
            metadata={"reading": reading},
        )

    def _handle_get_sensor_status(self, cmd: ParsedCommand) -> DispatchResult:
        status_dict = self.hardware.get_sensor_status()
        temp = status_dict.get("temperature")
        hum = status_dict.get("humidity")
        dist = status_dict.get("distance")

        is_sim = any(r.is_simulated for r in status_dict.values())
        prefix = "[Simulated] " if is_sim else ""

        text = (
            f"{prefix}Sensor diagnostic check: "
            f"Temperature is {temp.value:.1f} {temp.unit}, "
            f"Humidity is {hum.value:.0f} {hum.unit}, "
            f"Obstacle distance is {dist.value:.1f} {dist.unit}."
        )
        return DispatchResult(
            intent=cmd.intent,
            response_text=text,
            success=True,
            metadata=status_dict,
        )

    def _handle_open_door(self, cmd: ParsedCommand) -> DispatchResult:
        door_target = cmd.entities.get("door", "unspecified")
        if door_target == "unspecified":
            # Documented default behavior
            default_target = self.config.default_door
            res = self.hardware.open_door(default_target)
            prefix = "[Simulated] " if res.is_simulated else ""
            text = (
                f"{prefix}Opening {default_target} doors by default. "
                "You can also specify driver or passenger door."
            )
            return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)

        res = self.hardware.open_door(door_target)
        prefix = "[Simulated] " if res.is_simulated else ""
        text = f"{prefix}{res.message}"
        return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)

    def _handle_close_door(self, cmd: ParsedCommand) -> DispatchResult:
        condition = cmd.entities.get("condition")
        if condition == "both_open_only":
            door_status = self.hardware.get_door_status()
            driver_st = door_status.get("driver", "unknown")
            pass_st = door_status.get("passenger", "unknown")
            if driver_st == "unknown" or pass_st == "unknown":
                return DispatchResult(
                    intent=cmd.intent,
                    response_text="Cannot actuate: door position sensors are not installed and door state is unknown.",
                    success=False,
                )
            if driver_st == "open" and pass_st == "open":
                res = self.hardware.close_door("both")
                prefix = "[Simulated] " if res.is_simulated else ""
                text = f"{prefix}Both doors confirmed open. Closing both doors."
                return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)
            elif driver_st == "open" or pass_st == "open":
                open_which = "driver" if driver_st == "open" else "passenger"
                text = f"Condition not met: only the {open_which} door is open. Neither door was closed."
                return DispatchResult(intent=cmd.intent, response_text=text, success=True)
            else:
                text = "Both doors are already confirmed closed. No action taken."
                return DispatchResult(intent=cmd.intent, response_text=text, success=True)

        door_target = cmd.entities.get("door", "unspecified")
        if door_target == "unspecified":
            default_target = self.config.default_door
            res = self.hardware.close_door(default_target)
            prefix = "[Simulated] " if res.is_simulated else ""
            text = (
                f"{prefix}Closing {default_target} doors by default. "
                "You can also specify driver or passenger door."
            )
            return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)

        res = self.hardware.close_door(door_target)
        prefix = "[Simulated] " if res.is_simulated else ""
        text = f"{prefix}{res.message}"
        return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)

    def _handle_turn_ac_on(self, cmd: ParsedCommand) -> DispatchResult:
        res = self.hardware.set_ac(True)
        prefix = "[Simulated] " if res.is_simulated else ""
        text = f"{prefix}Air conditioning is now turned on."
        return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)

    def _handle_turn_ac_off(self, cmd: ParsedCommand) -> DispatchResult:
        res = self.hardware.set_ac(False)
        prefix = "[Simulated] " if res.is_simulated else ""
        text = f"{prefix}Air conditioning has been turned off."
        return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)

    def _handle_get_ac_status(self, cmd: ParsedCommand) -> DispatchResult:
        is_on = self.hardware.get_ac_status()
        state = "on" if is_on else "off"
        text = f"The air conditioning is currently {state}."
        return DispatchResult(intent=cmd.intent, response_text=text, success=True)

    def _handle_lights_on(self, cmd: ParsedCommand) -> DispatchResult:
        res = self.hardware.set_lights(True)
        prefix = "[Simulated] " if res.is_simulated else ""
        text = f"{prefix}Vehicle headlights turned on."
        return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)

    def _handle_lights_off(self, cmd: ParsedCommand) -> DispatchResult:
        res = self.hardware.set_lights(False)
        prefix = "[Simulated] " if res.is_simulated else ""
        text = f"{prefix}Vehicle headlights turned off."
        return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)

    def _handle_emergency_on(self, cmd: ParsedCommand) -> DispatchResult:
        res = self.hardware.set_emergency(True)
        prefix = "[Simulated] " if res.is_simulated else ""
        text = f"{prefix}Emergency hazard mode activated. All warning lights flashing."
        return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)

    def _handle_emergency_off(self, cmd: ParsedCommand) -> DispatchResult:
        res = self.hardware.set_emergency(False)
        prefix = "[Simulated] " if res.is_simulated else ""
        text = f"{prefix}Emergency hazard mode deactivated."
        return DispatchResult(intent=cmd.intent, response_text=text, success=res.success)

    def _handle_stop_listening(self, cmd: ParsedCommand) -> DispatchResult:
        return DispatchResult(
            intent=cmd.intent,
            response_text="Stopping listening mode.",
            success=True,
            should_stop_listening=True,
        )

    def _handle_help(self, cmd: ParsedCommand) -> DispatchResult:
        text = (
            "Available voice commands: check temperature, check humidity, "
            "obstacle distance, sensor diagnostics, turn AC on or off, "
            "check AC status, headlights on or off, open or close door, and exit."
        )
        return DispatchResult(intent=cmd.intent, response_text=text, success=True)

    def _handle_exit(self, cmd: ParsedCommand) -> DispatchResult:
        return DispatchResult(
            intent=cmd.intent,
            response_text="Goodbye! Vehicle assistant shutting down.",
            success=True,
            should_exit=True,
        )

    def _handle_get_lights_status(self, cmd: ParsedCommand) -> DispatchResult:
        is_on = self.hardware.get_lights_status()
        state = "on" if is_on else "off"
        text = f"Vehicle headlights are currently {state}."
        return DispatchResult(
            intent=cmd.intent,
            response_text=text,
            success=True,
            metadata={"lights_on": is_on},
        )

    def _handle_get_door_status(self, cmd: ParsedCommand) -> DispatchResult:
        doors = self.hardware.get_door_status()
        driver_state = doors.get("driver", "closed")
        pass_state = doors.get("passenger", "closed")
        text = f"Driver door is {driver_state} and passenger door is {pass_state}."
        return DispatchResult(
            intent=cmd.intent,
            response_text=text,
            success=True,
            metadata={"doors": doors},
        )

    def _handle_global_off(self, cmd: ParsedCommand) -> DispatchResult:
        res_lights = self.hardware.set_lights(False)
        res_ac = self.hardware.set_ac(False)
        if self.hardware.get_emergency_status():
            self.hardware.set_emergency(False)
        is_sim = res_lights.is_simulated or res_ac.is_simulated
        prefix = "[Simulated] " if is_sim else ""
        text = (
            f"{prefix}Switched off all controllable vehicle systems: headlights, "
            "air conditioning, and emergency hazard. Doors and safety systems were kept unchanged."
        )
        return DispatchResult(intent=cmd.intent, response_text=text, success=True)

    def _handle_global_on(self, cmd: ParsedCommand) -> DispatchResult:
        res_lights = self.hardware.set_lights(True)
        res_ac = self.hardware.set_ac(True)
        is_sim = res_lights.is_simulated or res_ac.is_simulated
        prefix = "[Simulated] " if is_sim else ""
        text = (
            f"{prefix}Switched on vehicle headlights and air conditioning. "
            "Doors and hazard mode were kept safe."
        )
        return DispatchResult(intent=cmd.intent, response_text=text, success=True)

    def _handle_clarify(self, cmd: ParsedCommand) -> DispatchResult:
        prompt = (cmd.plan and cmd.plan.clarification_prompt) or cmd.error_message or "Could you please clarify your request?"
        return DispatchResult(intent=cmd.intent, response_text=prompt, success=True)

    def _handle_ambiguous(self, cmd: ParsedCommand) -> DispatchResult:
        text = (cmd.plan and cmd.plan.clarification_prompt) or cmd.error_message or "Your request is ambiguous. Please specify driver door, passenger door, or both doors."
        return DispatchResult(
            intent=cmd.intent,
            response_text=text,
            success=False,
        )

    def _handle_compound(self, cmd: ParsedCommand) -> DispatchResult:
        """Executes an ordered sequence of actions, aggregates spoken responses, and handles partial failures."""
        results = []
        messages = []
        all_success = True
        should_exit = False
        should_stop_listening = False

        for action in cmd.actions:
            single_cmd = ParsedCommand(
                actions=[action],
                status=CommandStatus.VALID,
                raw_text=cmd.raw_text,
                normalized_text=cmd.normalized_text,
            )
            handler_name = f"_handle_{action.type.name.lower()}"
            handler = getattr(self, handler_name, None)
            if handler:
                res = handler(single_cmd)
                results.append(res)
                if not res.success:
                    all_success = False
                if res.should_exit:
                    should_exit = True
                if res.should_stop_listening:
                    should_stop_listening = True
                messages.append(res.response_text)
            else:
                all_success = False
                messages.append(f"Action {action.type.value} is not supported.")

        is_simulated = any(m.startswith("[Simulated] ") for m in messages)
        clean_msgs = []
        for m in messages:
            cleaned = m.replace("[Simulated] ", "").strip()
            if cleaned.endswith("."):
                cleaned = cleaned[:-1]
            clean_msgs.append(cleaned)

        if len(clean_msgs) == 1:
            body = clean_msgs[0]
        elif len(clean_msgs) == 2:
            first_c = clean_msgs[1][0].lower() if len(clean_msgs[1]) > 0 else ""
            rest_c = clean_msgs[1][1:] if len(clean_msgs[1]) > 1 else ""
            body = f"{clean_msgs[0]}, and {first_c}{rest_c}"
        else:
            first_parts = ", ".join(clean_msgs[:-1])
            last_msg = clean_msgs[-1]
            first_c = last_msg[0].lower() if len(last_msg) > 0 else ""
            rest_c = last_msg[1:] if len(last_msg) > 1 else ""
            body = f"{first_parts}, and {first_c}{rest_c}"

        prefix = "[Simulated] " if is_simulated else ""
        aggregated_text = f"{prefix}{body}."

        return DispatchResult(
            intent=IntentType.COMPOUND,
            response_text=aggregated_text,
            success=all_success,
            should_exit=should_exit,
            should_stop_listening=should_stop_listening,
            metadata={"action_results": [r.metadata for r in results]},
        )

    def _handle_unsupported(self, cmd: ParsedCommand) -> DispatchResult:
        return DispatchResult(
            intent=cmd.intent,
            response_text="Sorry, I did not recognize that command. Say 'help' for available options.",
            success=False,
        )
