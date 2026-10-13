"""Unit and integration tests for the Offline SVM Intent Classifier and Command Pipeline."""

import socket
import unittest
from pathlib import Path
from unittest.mock import patch

from app.assistant.dispatcher import CommandDispatcher
from app.hardware.simulation import SimulatedVehicleHardware
from app.intents.classifier import IntentClassifier
from app.intents.models import IntentType, CommandStatus, Action
from app.intents.svm_pipeline import SVMIntentPipeline


class TestSVMIntentClassifier(unittest.TestCase):
    """Test suite for offline SVM model, pipeline integration, and safety rules."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = IntentClassifier(enable_svm=True)
        cls.svm = cls.classifier.svm_pipeline
        cls.model_path = Path(__file__).resolve().parent.parent / "models" / "intent_svm.joblib"

    def test_svm_model_loaded(self):
        """Verifies that the trained SVM model is present and loaded with all 20 classes."""
        self.assertTrue(self.classifier.is_svm_active, "SVM classifier should be active.")
        self.assertIsNotNone(self.svm)
        self.assertTrue(self.svm.is_loaded)
        self.assertEqual(len(self.svm.classes), 23)

    def test_all_classes_recognized(self):
        """Verifies that all 23 classes can be recognized accurately."""
        class_probes = {
            IntentType.LIGHTS_ON: "turn on the headlights",
            IntentType.LIGHTS_OFF: "turn off the headlights",
            IntentType.GET_LIGHTS_STATUS: "are the headlights on",
            IntentType.TURN_AC_ON: "turn on the ac",
            IntentType.TURN_AC_OFF: "turn off the ac",
            IntentType.GET_AC_STATUS: "is the ac running",
            IntentType.OPEN_DOOR: "open the driver door",
            IntentType.CLOSE_DOOR: "close the driver door",
            IntentType.GET_DOOR_STATUS: "is the driver door open",
            IntentType.GET_TEMPERATURE: "what is the cabin temperature",
            IntentType.GET_HUMIDITY: "check the humidity",
            IntentType.GET_DISTANCE: "how far away is the obstacle",
            IntentType.GET_SENSOR_STATUS: "check all the sensors",
            IntentType.EMERGENCY_ON: "emergency hazard on",
            IntentType.EMERGENCY_OFF: "disable emergency mode",
            IntentType.GLOBAL_OFF: "turn off everything",
            IntentType.GLOBAL_ON: "turn everything on",
            IntentType.CLARIFY: "it is too dark in here",
            IntentType.GREETING: "hello assistant",
            IntentType.HELP: "help",
            IntentType.EXIT: "exit",
            IntentType.STOP_LISTENING: "stop listening",
            IntentType.UNSUPPORTED: "play some rock music on spotify",
        }

        for expected_intent, probe_text in class_probes.items():
            with self.subTest(intent=expected_intent):
                cmd = self.classifier.classify(probe_text)
                if expected_intent == IntentType.UNSUPPORTED:
                    self.assertIn(cmd.status, (CommandStatus.UNSUPPORTED, CommandStatus.REJECTED))
                else:
                    self.assertEqual(cmd.status, CommandStatus.VALID)
                    self.assertEqual(cmd.intent, expected_intent)

    def test_confusable_intent_pairs(self):
        """Verifies that opposite or closely-related intents are not confused."""
        pairs = [
            ("turn on headlights", IntentType.LIGHTS_ON, "turn off headlights", IntentType.LIGHTS_OFF),
            ("activate the ac", IntentType.TURN_AC_ON, "deactivate the ac", IntentType.TURN_AC_OFF),
            ("open driver door", IntentType.OPEN_DOOR, "close driver door", IntentType.CLOSE_DOOR),
            ("cabin temperature", IntentType.GET_TEMPERATURE, "cabin humidity", IntentType.GET_HUMIDITY),
            ("emergency on", IntentType.EMERGENCY_ON, "emergency off", IntentType.EMERGENCY_OFF),
        ]

        for text_a, intent_a, text_b, intent_b in pairs:
            cmd_a = self.classifier.classify(text_a)
            cmd_b = self.classifier.classify(text_b)
            self.assertEqual(cmd_a.intent, intent_a, f"Failed for {text_a}")
            self.assertEqual(cmd_b.intent, intent_b, f"Failed for {text_b}")

    def test_unsupported_and_out_of_domain_queries(self):
        """Verifies that out-of-scope requests are rejected as UNSUPPORTED."""
        unsupported_phrases = [
            "play music on spotify",
            "navigate to downtown",
            "what is the fastest route to work",
            "roll down the driver window",
            "roll up all windows",
            "open the sunroof",
            "adjust the side mirrors",
            "set temperature to 21 degrees",
            "lock the driver door",
            "unlock all doors",
        ]

        for phrase in unsupported_phrases:
            with self.subTest(phrase=phrase):
                cmd = self.classifier.classify(phrase)
                self.assertIn(
                    cmd.status,
                    (CommandStatus.UNSUPPORTED, CommandStatus.REJECTED),
                    f"'{phrase}' should be rejected as UNSUPPORTED or REJECTED",
                )

    def test_negation_and_negative_contrast(self):
        """Verifies that negative contrast sentences execute only the affirmative action."""
        cmd1 = self.classifier.classify("turn off the ac, not the headlights")
        self.assertEqual(cmd1.status, CommandStatus.VALID)
        self.assertEqual(len(cmd1.actions), 1)
        self.assertEqual(cmd1.actions[0].type, IntentType.TURN_AC_OFF)

        cmd2 = self.classifier.classify("turn on the headlights, not the ac")
        self.assertEqual(cmd2.status, CommandStatus.VALID)
        self.assertEqual(len(cmd2.actions), 1)
        self.assertEqual(cmd2.actions[0].type, IntentType.LIGHTS_ON)

    def test_compound_commands_two_and_three_actions(self):
        """Verifies that compound commands with multiple actions are parsed in order."""
        # Two actions
        cmd2 = self.classifier.classify("turn on the headlights and switch on the ac")
        self.assertEqual(cmd2.status, CommandStatus.VALID)
        self.assertEqual(len(cmd2.actions), 2)
        self.assertEqual(cmd2.actions[0].type, IntentType.LIGHTS_ON)
        self.assertEqual(cmd2.actions[1].type, IntentType.TURN_AC_ON)

        # Three actions
        cmd3 = self.classifier.classify("turn off the ac, switch off the headlights, and close both doors")
        self.assertEqual(cmd3.status, CommandStatus.VALID)
        self.assertEqual(len(cmd3.actions), 3)
        self.assertEqual(cmd3.actions[0].type, IntentType.TURN_AC_OFF)
        self.assertEqual(cmd3.actions[1].type, IntentType.LIGHTS_OFF)
        self.assertEqual(cmd3.actions[2].type, IntentType.CLOSE_DOOR)
        self.assertEqual(cmd3.actions[2].parameters.get("door"), "both")

        # Inversion order: "before opening driver door, turn off the ac"
        cmd_inv = self.classifier.classify("before opening driver door, turn off the ac")
        self.assertEqual(cmd_inv.status, CommandStatus.VALID)
        self.assertEqual(len(cmd_inv.actions), 2)
        self.assertEqual(cmd_inv.actions[0].type, IntentType.TURN_AC_OFF)
        self.assertEqual(cmd_inv.actions[1].type, IntentType.OPEN_DOOR)

    def test_conditional_door_closure_policy(self):
        """Verifies conditional door closure policy across all four environmental states."""
        utterance = "Close both doors if they are open, and don't close them if only one is open."
        cmd = self.classifier.classify(utterance)

        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(len(cmd.actions), 1)
        self.assertEqual(cmd.actions[0].type, IntentType.CLOSE_DOOR)
        self.assertEqual(cmd.actions[0].parameters.get("condition"), "both_open_only")

        hw = SimulatedVehicleHardware()
        dispatcher = CommandDispatcher(hw)

        # State 1: Both doors open -> Close both
        hw.doors = {"driver": "open", "passenger": "open"}
        res1 = dispatcher.dispatch(cmd)
        self.assertTrue(res1.success)
        self.assertEqual(hw.doors, {"driver": "closed", "passenger": "closed"})
        self.assertIn("Closing both doors", res1.response_text)

        # State 2: Exactly driver door open -> Do NOT close
        hw.doors = {"driver": "open", "passenger": "closed"}
        res2 = dispatcher.dispatch(cmd)
        self.assertTrue(res2.success)
        self.assertEqual(hw.doors, {"driver": "open", "passenger": "closed"})
        self.assertIn("only the driver door is open", res2.response_text)

        # State 3: Exactly passenger door open -> Do NOT close
        hw.doors = {"driver": "closed", "passenger": "open"}
        res3 = dispatcher.dispatch(cmd)
        self.assertTrue(res3.success)
        self.assertEqual(hw.doors, {"driver": "closed", "passenger": "open"})
        self.assertIn("only the passenger door is open", res3.response_text)

        # State 4: Both doors already closed -> Do nothing
        hw.doors = {"driver": "closed", "passenger": "closed"}
        res4 = dispatcher.dispatch(cmd)
        self.assertTrue(res4.success)
        self.assertEqual(hw.doors, {"driver": "closed", "passenger": "closed"})
        self.assertIn("already confirmed closed", res4.response_text)

        # State 5: Unknown door state -> Abort actuation
        hw.doors = {"driver": "unknown", "passenger": "unknown"}
        res5 = dispatcher.dispatch(cmd)
        self.assertFalse(res5.success)
        self.assertIn("unknown", res5.response_text)

    def test_missing_model_fallback(self):
        """Verifies that if the model file is missing, the system falls back safely to rules."""
        nonexistent_path = Path("/tmp/nonexistent_model_12345.joblib")
        fallback_clf = IntentClassifier(model_path=nonexistent_path, enable_svm=True)

        self.assertFalse(fallback_clf.is_svm_active)
        # Should still classify canonical commands using deterministic rules
        cmd = fallback_clf.classify("turn on the headlights")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(cmd.intent, IntentType.LIGHTS_ON)

    def test_save_and_reload_consistency(self):
        """Verifies that saving and reloading the pipeline reproduces identical predictions."""
        temp_path = Path("/tmp/test_intent_svm_reload.joblib")
        self.svm.save(temp_path)

        reloaded_svm = SVMIntentPipeline(temp_path)
        self.assertTrue(reloaded_svm.is_loaded)

        test_sentences = [
            "turn on the headlights",
            "shut off the ac",
            "open driver door",
            "what is the temperature",
        ]

        for s in test_sentences:
            pred_orig = self.svm.predict(s)
            pred_reloaded = reloaded_svm.predict(s)
            self.assertEqual(pred_orig.intent, pred_reloaded.intent)
            self.assertAlmostEqual(pred_orig.confidence, pred_reloaded.confidence, places=4)

        if temp_path.exists():
            temp_path.unlink()
        meta_temp = temp_path.with_name(temp_path.stem + "_meta.json")
        if meta_temp.exists():
            meta_temp.unlink()

    def test_offline_inference_makes_no_network_calls(self):
        """Verifies that running inference makes strictly zero network or socket calls."""
        def guarded_connect(*args, **kwargs):
            raise AssertionError("Network connection attempted during offline inference!")

        with patch.object(socket.socket, "connect", side_effect=guarded_connect):
            for phrase in ["turn on the lights", "open the door", "how hot is the cabin"]:
                cmd = self.classifier.classify(phrase)
                self.assertEqual(cmd.status, CommandStatus.VALID)


if __name__ == "__main__":
    unittest.main()
