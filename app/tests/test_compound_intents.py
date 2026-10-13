"""Comprehensive automated unit tests for compound commands and multi-action orchestration."""

import unittest
from typing import Dict

from app.assistant.dispatcher import CommandDispatcher
from app.hardware.interface import AbstractVehicleHardware, SensorReading, HardwareResponse
from app.hardware.simulation import SimulatedVehicleHardware
from app.intents.classifier import IntentClassifier
from app.intents.models import IntentType, CommandStatus


class MockFaultyHardware(AbstractVehicleHardware):
    """Faulty hardware that fails only door actuations to test partial failure handling."""

    def __init__(self):
        self.lights_on = False
        self.ac_on = False

    def get_temperature(self) -> SensorReading:
        return SensorReading(value=22.0, unit="°C", is_simulated=True, status="OK")

    def get_humidity(self) -> SensorReading:
        return SensorReading(value=50.0, unit="%", is_simulated=True, status="OK")

    def get_distance(self) -> SensorReading:
        return SensorReading(value=100.0, unit="cm", is_simulated=True, status="OK")

    def get_sensor_status(self) -> Dict[str, SensorReading]:
        return {}

    def set_lights(self, enable: bool) -> HardwareResponse:
        self.lights_on = enable
        return HardwareResponse(success=True, message=f"Headlights switched {'ON' if enable else 'OFF'}", is_simulated=True)

    def get_lights_status(self) -> bool:
        return self.lights_on

    def set_ac(self, enable: bool) -> HardwareResponse:
        self.ac_on = enable
        return HardwareResponse(success=True, message=f"AC switched {'ON' if enable else 'OFF'}", is_simulated=True)

    def get_ac_status(self) -> bool:
        return self.ac_on

    def set_emergency(self, enable: bool) -> HardwareResponse:
        return HardwareResponse(success=True, message="Emergency mode changed", is_simulated=True)

    def get_emergency_status(self) -> bool:
        return False

    def open_door(self, door: str) -> HardwareResponse:
        return HardwareResponse(success=False, message="Simulated servo stall on door open", is_simulated=True)

    def close_door(self, door: str) -> HardwareResponse:
        return HardwareResponse(success=False, message="Simulated servo stall on door close", is_simulated=True)

    def get_door_status(self) -> Dict[str, str]:
        return {"driver": "error", "passenger": "error"}

    def cleanup(self) -> None:
        pass


class TestCompoundVoiceCommands(unittest.TestCase):
    """Verifies deterministic parsing and sequential dispatch of compound utterances."""

    def setUp(self):
        self.classifier = IntentClassifier()
        self.sim_hw = SimulatedVehicleHardware()
        self.dispatcher = CommandDispatcher(self.sim_hw)

    def test_compound_objects_distribution(self):
        """Verifies single verb distributing over multiple device objects."""
        # AC and headlights ON
        cmd1 = self.classifier.classify("turn on the ac and headlights")
        self.assertEqual(cmd1.status, CommandStatus.VALID)
        self.assertEqual(cmd1.intent, IntentType.COMPOUND)
        self.assertEqual([a.type for a in cmd1.actions], [IntentType.TURN_AC_ON, IntentType.LIGHTS_ON])

        res1 = self.dispatcher.dispatch(cmd1)
        self.assertTrue(res1.success)
        self.assertTrue(self.sim_hw.get_ac_status())
        self.assertTrue(self.sim_hw.get_lights_status())
        self.assertIn("turned on", res1.response_text.lower())

        # AC and headlights OFF
        cmd2 = self.classifier.classify("turn off both the ac and the lights")
        self.assertEqual(cmd2.status, CommandStatus.VALID)
        self.assertEqual([a.type for a in cmd2.actions], [IntentType.TURN_AC_OFF, IntentType.LIGHTS_OFF])

        res2 = self.dispatcher.dispatch(cmd2)
        self.assertTrue(res2.success)
        self.assertFalse(self.sim_hw.get_ac_status())
        self.assertFalse(self.sim_hw.get_lights_status())

    def test_sequential_conjunctions_and_ordering(self):
        """Verifies sequential conjunctions and multi-verb command ordering."""
        cmd = self.classifier.classify("open the driver door and turn on the headlights")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(len(cmd.actions), 2)
        self.assertEqual(cmd.actions[0].type, IntentType.OPEN_DOOR)
        self.assertEqual(cmd.actions[0].parameters.get("door"), "driver")
        self.assertEqual(cmd.actions[1].type, IntentType.LIGHTS_ON)

        res = self.dispatcher.dispatch(cmd)
        self.assertTrue(res.success)
        self.assertEqual(self.sim_hw.doors["driver"], "open")
        self.assertTrue(self.sim_hw.get_lights_status())
        self.assertIn("driver door opened", res.response_text.lower())
        self.assertIn("headlights turned on", res.response_text.lower())

    def test_inverted_ordering_with_before(self):
        """Verifies 'before A, B' inverts execution order: B executes first, then A."""
        cmd = self.classifier.classify("before opening the passenger door, turn off the ac")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(len(cmd.actions), 2)
        self.assertEqual(cmd.actions[0].type, IntentType.TURN_AC_OFF)
        self.assertEqual(cmd.actions[1].type, IntentType.OPEN_DOOR)
        self.assertEqual(cmd.actions[1].parameters.get("door"), "passenger")

    def test_pronoun_anaphora_resolution(self):
        """Verifies resolution of 'them' and 'it' referring to earlier devices."""
        cmd_lights = self.classifier.classify("turn on the headlights, then turn them off")
        self.assertEqual(cmd_lights.status, CommandStatus.VALID)
        self.assertEqual([a.type for a in cmd_lights.actions], [IntentType.LIGHTS_ON, IntentType.LIGHTS_OFF])

        cmd_ac = self.classifier.classify("start the ac, then turn it off")
        self.assertEqual(cmd_ac.status, CommandStatus.VALID)
        self.assertEqual([a.type for a in cmd_ac.actions], [IntentType.TURN_AC_ON, IntentType.TURN_AC_OFF])

    def test_contrast_and_state_preservation(self):
        """Verifies 'keep ... on' and 'leave ... off' contrast statements."""
        cmd = self.classifier.classify("turn off the headlights but keep the ac on")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual([a.type for a in cmd.actions], [IntentType.LIGHTS_OFF, IntentType.TURN_AC_ON])

        res = self.dispatcher.dispatch(cmd)
        self.assertTrue(res.success)
        self.assertFalse(self.sim_hw.get_lights_status())
        self.assertTrue(self.sim_hw.get_ac_status())

    def test_negative_contrast(self):
        """Verifies 'turn off the ac, not the headlights' executes only affirmative action."""
        cmd = self.classifier.classify("turn off the ac, not the headlights")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual([a.type for a in cmd.actions], [IntentType.TURN_AC_OFF])

    def test_three_action_compound_commands(self):
        """Verifies 3-action compound utterances."""
        cmd1 = self.classifier.classify("turn on the headlights, ac, and open the driver door")
        self.assertEqual(cmd1.status, CommandStatus.VALID)
        self.assertEqual(
            [a.type for a in cmd1.actions],
            [IntentType.LIGHTS_ON, IntentType.TURN_AC_ON, IntentType.OPEN_DOOR],
        )
        self.assertEqual(cmd1.actions[2].parameters.get("door"), "driver")

        res1 = self.dispatcher.dispatch(cmd1)
        self.assertTrue(res1.success)
        self.assertTrue(self.sim_hw.get_lights_status())
        self.assertTrue(self.sim_hw.get_ac_status())
        self.assertEqual(self.sim_hw.doors["driver"], "open")

        cmd2 = self.classifier.classify("turn off the ac, switch off the headlights, and close both doors")
        self.assertEqual(cmd2.status, CommandStatus.VALID)
        self.assertEqual(
            [a.type for a in cmd2.actions],
            [IntentType.TURN_AC_OFF, IntentType.LIGHTS_OFF, IntentType.CLOSE_DOOR],
        )
        self.assertEqual(cmd2.actions[2].parameters.get("door"), "both")

        res2 = self.dispatcher.dispatch(cmd2)
        self.assertTrue(res2.success)
        self.assertFalse(self.sim_hw.get_ac_status())
        self.assertFalse(self.sim_hw.get_lights_status())
        self.assertEqual(self.sim_hw.doors["driver"], "closed")
        self.assertEqual(self.sim_hw.doors["passenger"], "closed")

    def test_safety_rejection_mixed_supported_unsupported(self):
        """Verifies compound command is rejected if ANY sub-action is unsupported."""
        cmd = self.classifier.classify("turn off the ac and play rock music")
        self.assertEqual(cmd.status, CommandStatus.REJECTED)
        self.assertEqual(len(cmd.actions), 0)

        # Dispatcher must reject without executing any partial action
        self.sim_hw.set_ac(True)
        res = self.dispatcher.dispatch(cmd)
        self.assertFalse(res.success)
        self.assertIn("rejected", res.response_text.lower())
        # AC state should remain untouched (not turned off)
        self.assertTrue(self.sim_hw.get_ac_status())

    def test_safety_rejection_ambiguous_door_in_compound(self):
        """Verifies compound command with ambiguous door target is rejected."""
        cmd = self.classifier.classify("open the door and turn on the ac")
        self.assertEqual(cmd.status, CommandStatus.REJECTED)
        self.assertEqual(len(cmd.actions), 0)

        res = self.dispatcher.dispatch(cmd)
        self.assertFalse(res.success)
        self.assertIn("rejected", res.response_text.lower())

    def test_partial_failure_handling_in_dispatcher(self):
        """Verifies dispatcher reports partial failure when an action in sequence fails."""
        faulty_hw = MockFaultyHardware()
        dispatcher = CommandDispatcher(faulty_hw)

        cmd = self.classifier.classify("turn on the headlights and open the driver door")
        self.assertEqual(len(cmd.actions), 2)

        res = dispatcher.dispatch(cmd)
        # One action succeeded (lights), one action failed (servo stall)
        self.assertFalse(res.success)
        self.assertTrue(faulty_hw.get_lights_status())
        self.assertIn("stall", res.response_text.lower())


if __name__ == "__main__":
    unittest.main()
