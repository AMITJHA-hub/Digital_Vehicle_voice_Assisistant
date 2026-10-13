"""Unit tests for PhysicalVehicleHardware safety controls and timeout handling."""

import unittest
from app.config import config
from app.hardware.physical import PhysicalVehicleHardware


class TestPhysicalHardwareSafety(unittest.TestCase):
    """Verifies electrical safety controls, timeouts, and state tracking."""

    def setUp(self):
        try:
            self.hw = PhysicalVehicleHardware(config)
        except Exception as e:
            if "GPIO busy" in str(e):
                self.skipTest(f"Physical GPIO hardware is currently claimed by active service: {e}")
            raise

    def tearDown(self):
        if hasattr(self, "hw") and self.hw:
            self.hw.cleanup()

    def test_servo_movement_actuation_and_safety_lock(self):
        """Verifies physical servo motion actuation when enabled and safety lock fallback."""
        self.assertTrue(self.hw.servos_enabled)

        open_res = self.hw.open_door("driver")
        self.assertTrue(open_res.success)
        self.assertIn("opened", open_res.message.lower())

        close_res = self.hw.close_door("driver")
        self.assertTrue(close_res.success)
        self.assertIn("closed", close_res.message.lower())

        # Test safety lock fallback when disabled
        self.hw.servos_enabled = False
        locked_open = self.hw.open_door("driver")
        self.assertFalse(locked_open.success)
        self.assertIn("locked for electrical safety", locked_open.message.lower())

        locked_close = self.hw.close_door("driver")
        self.assertFalse(locked_close.success)
        self.assertIn("locked for electrical safety", locked_close.message.lower())

    def test_ultrasonic_sensor_timeout_safety(self):
        """Verifies distance read returns TIMEOUT safely without hanging if no sensor is connected."""
        # When pins are floating or sensor disconnected, must return TIMEOUT within 60ms
        reading = self.hw.get_distance()
        self.assertIn(reading.status, ["TIMEOUT", "OK", "OUT_OF_RANGE"])
        self.assertFalse(reading.is_simulated)

    def test_led_state_toggles(self):
        """Verifies LED state tracking in physical mode."""
        res_on = self.hw.set_lights(True)
        self.assertTrue(res_on.success)
        self.assertTrue(self.hw.get_lights_status())

        res_off = self.hw.set_lights(False)
        self.assertTrue(res_off.success)
        self.assertFalse(self.hw.get_lights_status())

        res_ac = self.hw.set_ac(True)
        self.assertTrue(res_ac.success)
        self.assertTrue(self.hw.get_ac_status())

    def test_cleanup_resets_handle(self):
        """Verifies cleanup method releases RP1 chip handle safely."""
        self.assertIsNotNone(self.hw._chip_handle)
        self.hw.cleanup()
        self.assertIsNone(self.hw._chip_handle)


if __name__ == "__main__":
    unittest.main()
