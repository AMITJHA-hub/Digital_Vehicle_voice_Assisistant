"""Automated unit test suite for intent classifier and command dispatcher."""

import unittest
from typing import Dict

from app.config import AppConfig
from app.assistant.dispatcher import CommandDispatcher
from app.hardware.interface import AbstractVehicleHardware, SensorReading, HardwareResponse
from app.hardware.simulation import SimulatedVehicleHardware
from app.intents.classifier import IntentClassifier
from app.intents.models import IntentType


class MockFaultyHardware(AbstractVehicleHardware):
    """Simulated hardware that triggers sensor failures and timeouts for robustness tests."""

    def get_temperature(self) -> SensorReading:
        return SensorReading(value=0.0, unit="°C", is_simulated=True, status="ERROR")

    def get_humidity(self) -> SensorReading:
        return SensorReading(value=0.0, unit="%", is_simulated=True, status="TIMEOUT")

    def get_distance(self) -> SensorReading:
        return SensorReading(value=0.0, unit="cm", is_simulated=True, status="DISCONNECTED")

    def get_sensor_status(self) -> Dict[str, SensorReading]:
        return {
            "temperature": self.get_temperature(),
            "humidity": self.get_humidity(),
            "distance": self.get_distance(),
        }

    def set_lights(self, enable: bool) -> HardwareResponse:
        return HardwareResponse(success=False, message="Simulated relay failure", is_simulated=True)

    def get_lights_status(self) -> bool:
        return False

    def set_ac(self, enable: bool) -> HardwareResponse:
        return HardwareResponse(success=False, message="Simulated AC clutch failure", is_simulated=True)

    def get_ac_status(self) -> bool:
        return False

    def set_emergency(self, enable: bool) -> HardwareResponse:
        return HardwareResponse(success=False, message="Simulated flasher failure", is_simulated=True)

    def get_emergency_status(self) -> bool:
        return False

    def open_door(self, door: str) -> HardwareResponse:
        return HardwareResponse(success=False, message="Simulated servo stall", is_simulated=True)

    def close_door(self, door: str) -> HardwareResponse:
        return HardwareResponse(success=False, message="Simulated servo stall", is_simulated=True)

    def get_door_status(self) -> Dict[str, str]:
        return {"driver": "error", "passenger": "error"}

    def cleanup(self) -> None:
        pass


class TestIntentClassificationAndDispatch(unittest.TestCase):
    """Verifies deterministic intent parsing and dispatcher orchestration."""

    def setUp(self):
        self.classifier = IntentClassifier()
        self.sim_hw = SimulatedVehicleHardware()
        self.dispatcher = CommandDispatcher(self.sim_hw)

    def test_all_supported_intents(self):
        """Verifies every supported intent is recognized by at least one canonical query."""
        queries = {
            "hello": IntentType.GREETING,
            "what is the cabin temperature": IntentType.GET_TEMPERATURE,
            "what is the humidity": IntentType.GET_HUMIDITY,
            "how far is the obstacle": IntentType.GET_DISTANCE,
            "check the sensors": IntentType.GET_SENSOR_STATUS,
            "open the door": IntentType.OPEN_DOOR,
            "close the door": IntentType.CLOSE_DOOR,
            "turn the ac on": IntentType.TURN_AC_ON,
            "turn the ac off": IntentType.TURN_AC_OFF,
            "what is the ac status": IntentType.GET_AC_STATUS,
            "turn on the lights": IntentType.LIGHTS_ON,
            "turn off the lights": IntentType.LIGHTS_OFF,
            "emergency mode": IntentType.EMERGENCY_ON,
            "emergency off": IntentType.EMERGENCY_OFF,
            "stop listening": IntentType.STOP_LISTENING,
            "help": IntentType.HELP,
            "exit": IntentType.EXIT,
            "play radio music": IntentType.UNSUPPORTED,
        }

        for phrase, expected_intent in queries.items():
            cmd = self.classifier.classify(phrase)
            self.assertEqual(
                cmd.intent,
                expected_intent,
                f"Failed for phrase '{phrase}': expected {expected_intent}, got {cmd.intent}",
            )

    def test_different_phrasings_same_intent(self):
        """Verifies multiple natural variations map correctly to the same intent."""
        # Lights
        self.assertEqual(self.classifier.classify("switch on the lights").intent, IntentType.LIGHTS_ON)
        self.assertEqual(self.classifier.classify("turn headlights on").intent, IntentType.LIGHTS_ON)
        self.assertEqual(self.classifier.classify("lights off").intent, IntentType.LIGHTS_OFF)
        self.assertEqual(self.classifier.classify("turn off headlights").intent, IntentType.LIGHTS_OFF)

        # Temperature
        self.assertEqual(self.classifier.classify("how hot is the cabin").intent, IntentType.GET_TEMPERATURE)
        self.assertEqual(self.classifier.classify("current temperature").intent, IntentType.GET_TEMPERATURE)
        self.assertEqual(self.classifier.classify("what's the temp in the car").intent, IntentType.GET_TEMPERATURE)

        # AC
        self.assertEqual(self.classifier.classify("turn air conditioning on").intent, IntentType.TURN_AC_ON)
        self.assertEqual(self.classifier.classify("switch off ac").intent, IntentType.TURN_AC_OFF)
        self.assertEqual(self.classifier.classify("is the ac on").intent, IntentType.GET_AC_STATUS)

        # Distance
        self.assertEqual(self.classifier.classify("how close is the wall").intent, IntentType.GET_DISTANCE)
        self.assertEqual(self.classifier.classify("range to object").intent, IntentType.GET_DISTANCE)

    def test_empty_and_unrecognized_transcripts(self):
        """Verifies empty or nonsense transcripts return UNSUPPORTED safely."""
        self.assertEqual(self.classifier.classify("").intent, IntentType.UNSUPPORTED)
        self.assertEqual(self.classifier.classify("   ").intent, IntentType.UNSUPPORTED)
        self.assertEqual(self.classifier.classify("blablabla 12345").intent, IntentType.UNSUPPORTED)

    def test_door_entity_extraction_and_ambiguity(self):
        """Verifies driver, passenger, both, and unspecified door handling."""
        # Driver door
        cmd = self.classifier.classify("open the driver door")
        self.assertEqual(cmd.intent, IntentType.OPEN_DOOR)
        self.assertEqual(cmd.entities.get("door"), "driver")
        res = self.dispatcher.dispatch(cmd)
        self.assertIn("driver door opened", res.response_text.lower())
        self.assertEqual(self.sim_hw.doors["driver"], "open")

        # Passenger door
        cmd = self.classifier.classify("close passenger door")
        self.assertEqual(cmd.intent, IntentType.CLOSE_DOOR)
        self.assertEqual(cmd.entities.get("door"), "passenger")
        res = self.dispatcher.dispatch(cmd)
        self.assertIn("passenger door closed", res.response_text.lower())

        # Both doors
        cmd = self.classifier.classify("open both doors")
        self.assertEqual(cmd.intent, IntentType.OPEN_DOOR)
        self.assertEqual(cmd.entities.get("door"), "both")

        # Ambiguous door ("open the door" without specifying)
        cmd_ambiguous = self.classifier.classify("open the door")
        self.assertEqual(cmd_ambiguous.intent, IntentType.OPEN_DOOR)
        self.assertEqual(cmd_ambiguous.entities.get("door"), "unspecified")
        res = self.dispatcher.dispatch(cmd_ambiguous)
        # Verify documented default behavior is explained to the user
        self.assertIn("default", res.response_text.lower())
        self.assertIn("specify", res.response_text.lower())

    def test_state_transitions_lights_and_ac(self):
        """Verifies state updates in hardware for lights and AC."""
        # Initial state
        self.assertFalse(self.sim_hw.get_lights_status())
        self.assertFalse(self.sim_hw.get_ac_status())

        # Turn lights on
        cmd = self.classifier.classify("turn on the lights")
        res = self.dispatcher.dispatch(cmd)
        self.assertTrue(res.success)
        self.assertTrue(self.sim_hw.get_lights_status())

        # Turn lights off
        cmd = self.classifier.classify("turn off the lights")
        res = self.dispatcher.dispatch(cmd)
        self.assertTrue(res.success)
        self.assertFalse(self.sim_hw.get_lights_status())

        # Turn AC on
        cmd = self.classifier.classify("turn the ac on")
        res = self.dispatcher.dispatch(cmd)
        self.assertTrue(res.success)
        self.assertTrue(self.sim_hw.get_ac_status())

        # Check AC status
        cmd = self.classifier.classify("what is the ac status")
        res = self.dispatcher.dispatch(cmd)
        self.assertIn("currently on", res.response_text)

        # Turn AC off
        cmd = self.classifier.classify("turn the ac off")
        res = self.dispatcher.dispatch(cmd)
        self.assertFalse(self.sim_hw.get_ac_status())

    def test_simulation_labeling(self):
        """Verifies that simulation responses are explicitly marked as [Simulated]."""
        cmd = self.classifier.classify("what is the cabin temperature")
        res = self.dispatcher.dispatch(cmd)
        self.assertTrue(res.response_text.startswith("[Simulated]"))
        self.assertIn("23.5", res.response_text)

        cmd = self.classifier.classify("how far is the obstacle")
        res = self.dispatcher.dispatch(cmd)
        self.assertTrue(res.response_text.startswith("[Simulated]"))
        self.assertIn("85.0", res.response_text)

    def test_faulty_sensor_handling(self):
        """Verifies graceful response when sensors error or timeout."""
        faulty_hw = MockFaultyHardware()
        faulty_dispatcher = CommandDispatcher(faulty_hw)

        cmd_temp = self.classifier.classify("what is the temperature")
        res_temp = faulty_dispatcher.dispatch(cmd_temp)
        self.assertFalse(res_temp.success)
        self.assertIn("ERROR", res_temp.response_text)

        cmd_hum = self.classifier.classify("check humidity")
        res_hum = faulty_dispatcher.dispatch(cmd_hum)
        self.assertFalse(res_hum.success)
        self.assertIn("TIMEOUT", res_hum.response_text)

        cmd_dist = self.classifier.classify("how far is the obstacle")
        res_dist = faulty_dispatcher.dispatch(cmd_dist)
        self.assertFalse(res_dist.success)
        self.assertIn("DISCONNECTED", res_dist.response_text)

    def test_noisy_transcripts_from_whisper(self):
        """Verifies handling of transcripts containing whisper annotations or noise."""
        raw_noisy = "  what's the cabin temperature please. "
        cmd = self.classifier.classify(raw_noisy)
        self.assertEqual(cmd.intent, IntentType.GET_TEMPERATURE)

        raw_ac = "... turn the ac on now ..."
        cmd = self.classifier.classify(raw_ac)
        self.assertEqual(cmd.intent, IntentType.TURN_AC_ON)

    def test_simulation_default_safety(self):
        """Verifies default config strictly prevents unintentional physical hardware access."""
        default_cfg = AppConfig()
        self.assertEqual(default_cfg.hardware_mode, "simulation")


if __name__ == "__main__":
    unittest.main()
