"""Regression tests for expanded natural-language, collective, and implicit vehicle commands."""

import unittest

from app.hardware.simulation import SimulatedVehicleHardware
from app.intents.classifier import IntentClassifier
from app.intents.models import IntentType, CommandStatus, Action
from app.assistant.dispatcher import CommandDispatcher


class TestNaturalLanguageCommands(unittest.TestCase):
    """Verifies all 15 required natural-language and implicit command capabilities."""

    def setUp(self):
        self.hardware = SimulatedVehicleHardware()
        self.classifier = IntentClassifier(enable_svm=True)
        self.dispatcher = CommandDispatcher(self.hardware)
        self.classifier.reset_context()

    # 1. "Turn off everything."
    def test_01_turn_off_everything(self):
        # Pre-condition: turn on lights, AC, and emergency hazard
        self.hardware.set_lights(True)
        self.hardware.set_ac(True)
        self.hardware.set_emergency(True)
        self.hardware.open_door("driver")

        cmd = self.classifier.classify("Turn off everything.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(cmd.intent, IntentType.GLOBAL_OFF)

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        self.assertFalse(self.hardware.get_lights_status())
        self.assertFalse(self.hardware.get_ac_status())
        self.assertFalse(self.hardware.get_emergency_status())
        # SAFETY: Doors must remain untouched by global commands
        self.assertEqual(self.hardware.get_door_status()["driver"], "open")

    # 2. "Switch everything off."
    def test_02_switch_everything_off(self):
        self.hardware.set_lights(True)
        self.hardware.set_ac(True)

        cmd = self.classifier.classify("Switch everything off.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(cmd.intent, IntentType.GLOBAL_OFF)

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        self.assertFalse(self.hardware.get_lights_status())
        self.assertFalse(self.hardware.get_ac_status())

    # 3. "Turn everything on."
    def test_03_turn_everything_on(self):
        self.hardware.set_lights(False)
        self.hardware.set_ac(False)
        self.hardware.close_door("both")

        cmd = self.classifier.classify("Turn everything on.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(cmd.intent, IntentType.GLOBAL_ON)

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        # Safe primary accessories activated
        self.assertTrue(self.hardware.get_lights_status())
        self.assertTrue(self.hardware.get_ac_status())
        # SAFETY: Doors and hazard are NOT actuated by "everything on"
        self.assertEqual(self.hardware.get_door_status()["driver"], "closed")
        self.assertEqual(self.hardware.get_door_status()["passenger"], "closed")
        self.assertFalse(self.hardware.get_emergency_status())

    # 4. "Turn off everything except the headlights."
    def test_04_turn_off_everything_except_headlights(self):
        self.hardware.set_lights(True)
        self.hardware.set_ac(True)
        self.hardware.set_emergency(True)

        cmd = self.classifier.classify("Turn off everything except the headlights.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertIn("headlights", cmd.plan.excluded_targets)
        action_types = [a.type for a in cmd.actions]
        self.assertIn(IntentType.TURN_AC_OFF, action_types)
        self.assertIn(IntentType.EMERGENCY_OFF, action_types)
        self.assertNotIn(IntentType.LIGHTS_OFF, action_types)

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        self.assertTrue(self.hardware.get_lights_status(), "Headlights must remain ON!")
        self.assertFalse(self.hardware.get_ac_status())
        self.assertFalse(self.hardware.get_emergency_status())

    # 5. "Turn off the AC, not the headlights."
    def test_05_turn_off_ac_not_headlights(self):
        self.hardware.set_lights(True)
        self.hardware.set_ac(True)

        cmd = self.classifier.classify("Turn off the AC, not the headlights.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(len(cmd.actions), 1)
        self.assertEqual(cmd.actions[0].type, IntentType.TURN_AC_OFF)
        self.assertIn("headlights", cmd.plan.excluded_targets)

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        self.assertTrue(self.hardware.get_lights_status(), "Headlights must remain ON!")
        self.assertFalse(self.hardware.get_ac_status())

    # 6. "Turn on the headlights and AC."
    def test_06_turn_on_headlights_and_ac(self):
        self.hardware.set_lights(False)
        self.hardware.set_ac(False)

        cmd = self.classifier.classify("Turn on the headlights and AC.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(len(cmd.actions), 2)
        self.assertEqual(cmd.actions[0].type, IntentType.LIGHTS_ON)
        self.assertEqual(cmd.actions[1].type, IntentType.TURN_AC_ON)

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        self.assertTrue(self.hardware.get_lights_status())
        self.assertTrue(self.hardware.get_ac_status())

    # 7. "Turn on the lights and leave them on."
    def test_07_turn_on_lights_and_leave_them_on(self):
        self.hardware.set_lights(False)

        cmd = self.classifier.classify("Turn on the lights and leave them on.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(len(cmd.actions), 1)
        self.assertEqual(cmd.actions[0].type, IntentType.LIGHTS_ON)

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        self.assertTrue(self.hardware.get_lights_status())

    # 8. "Turn on the AC and then turn it off."
    def test_08_turn_on_ac_and_then_turn_it_off(self):
        self.hardware.set_ac(False)

        cmd = self.classifier.classify("Turn on the AC and then turn it off.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(len(cmd.actions), 2)
        self.assertEqual(cmd.actions[0].type, IntentType.TURN_AC_ON)
        self.assertEqual(cmd.actions[1].type, IntentType.TURN_AC_OFF)

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        # End simulated state is turned OFF
        self.assertFalse(self.hardware.get_ac_status())

    # 9. "Open the driver door and close it."
    def test_09_open_driver_door_and_close_it(self):
        self.hardware.close_door("both")

        cmd = self.classifier.classify("Open the driver door and close it.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(len(cmd.actions), 2)
        self.assertEqual(cmd.actions[0].type, IntentType.OPEN_DOOR)
        self.assertEqual(cmd.actions[0].parameters["door"], "driver")
        self.assertEqual(cmd.actions[1].type, IntentType.CLOSE_DOOR)
        self.assertEqual(cmd.actions[1].parameters["door"], "driver")

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        self.assertEqual(self.hardware.get_door_status()["driver"], "closed")

    # 10. "Don't touch the doors; turn the lights off."
    def test_10_dont_touch_doors_turn_lights_off(self):
        self.hardware.set_lights(True)
        self.hardware.open_door("driver")

        cmd = self.classifier.classify("Don't touch the doors; turn the lights off.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertIn("doors", cmd.plan.excluded_targets)
        self.assertEqual(len(cmd.actions), 1)
        self.assertEqual(cmd.actions[0].type, IntentType.LIGHTS_OFF)

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        self.assertFalse(self.hardware.get_lights_status())
        self.assertEqual(self.hardware.get_door_status()["driver"], "open", "Driver door must remain open!")

    # 11. "Turn the AC off but keep it running."
    def test_11_turn_ac_off_but_keep_it_running_contradiction(self):
        self.hardware.set_ac(True)

        cmd = self.classifier.classify("Turn the AC off but keep it running.")
        self.assertIn(cmd.status, (CommandStatus.REJECTED, CommandStatus.AMBIGUOUS))
        self.assertTrue(cmd.plan.is_contradictory)
        self.assertIn("Contradictory", cmd.plan.clarification_prompt)

        disp = self.dispatcher.dispatch(cmd)
        self.assertFalse(disp.success)
        # SAFETY: Hardware state unchanged
        self.assertTrue(self.hardware.get_ac_status())

    # 12. "It's too dark in here."
    def test_12_its_too_dark_in_here_clarification(self):
        self.hardware.set_lights(False)

        cmd = self.classifier.classify("It's too dark in here.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(cmd.intent, IntentType.CLARIFY)
        self.assertEqual(cmd.entities.get("target"), "headlights")

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        self.assertIn("Would you like me to turn on the headlights", disp.response_text)
        # CRITICAL: No automatic actuation
        self.assertFalse(self.hardware.get_lights_status())

    # 13. "Make the cabin more comfortable."
    def test_13_make_cabin_more_comfortable_clarification(self):
        self.hardware.set_ac(False)

        cmd = self.classifier.classify("Make the cabin more comfortable.")
        self.assertEqual(cmd.status, CommandStatus.VALID)
        self.assertEqual(cmd.intent, IntentType.CLARIFY)
        self.assertEqual(cmd.entities.get("target"), "comfort")

        disp = self.dispatcher.dispatch(cmd)
        self.assertTrue(disp.success)
        self.assertIn("air conditioning", disp.response_text)
        self.assertFalse(self.hardware.get_ac_status())

    # 14. "Switch both of those off" with and without valid prior context
    def test_14_switch_both_of_those_off_context_handling(self):
        # A. WITHOUT valid prior context (clean initial state)
        self.classifier.reset_context()
        cmd_no_ctx = self.classifier.classify("Switch both of those off.")
        self.assertEqual(cmd_no_ctx.status, CommandStatus.AMBIGUOUS)
        self.assertIn("Cannot determine", cmd_no_ctx.plan.clarification_prompt)

        disp_no_ctx = self.dispatcher.dispatch(cmd_no_ctx)
        self.assertFalse(disp_no_ctx.success)

        # B. WITH valid prior context
        self.hardware.set_lights(False)
        self.hardware.set_ac(False)

        # Antecedent command
        cmd_antecedent = self.classifier.classify("Turn on the headlights and AC.")
        self.dispatcher.dispatch(cmd_antecedent)
        self.assertTrue(self.hardware.get_lights_status())
        self.assertTrue(self.hardware.get_ac_status())

        # Follow-up elliptical command
        cmd_followup = self.classifier.classify("Switch both of those off.")
        self.assertEqual(cmd_followup.status, CommandStatus.VALID)
        action_types = [a.type for a in cmd_followup.actions]
        self.assertIn(IntentType.LIGHTS_OFF, action_types)
        self.assertIn(IntentType.TURN_AC_OFF, action_types)

        disp_followup = self.dispatcher.dispatch(cmd_followup)
        self.assertTrue(disp_followup.success)
        self.assertFalse(self.hardware.get_lights_status())
        self.assertFalse(self.hardware.get_ac_status())

    # 15. Unsupported command that must not trigger any action
    def test_15_unsupported_command_no_action(self):
        self.hardware.set_lights(False)
        self.hardware.set_ac(False)

        cmd = self.classifier.classify("Roll down the windows and play jazz music on Spotify.")
        self.assertIn(cmd.status, (CommandStatus.UNSUPPORTED, CommandStatus.REJECTED))
        self.assertEqual(len(cmd.actions), 0)

        disp = self.dispatcher.dispatch(cmd)
        self.assertFalse(disp.success)
        self.assertFalse(self.hardware.get_lights_status())
        self.assertFalse(self.hardware.get_ac_status())


if __name__ == "__main__":
    unittest.main()
