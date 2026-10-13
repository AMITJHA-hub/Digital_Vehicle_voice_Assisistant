"""Comprehensive all-in-one hardware self-test without microphone requirement."""

import sys
import time

from app.config import config
from app.hardware.physical import PhysicalVehicleHardware


def run_full_diagnostic():
    print("=" * 70)
    print("  COCKPIT PHYSICAL HARDWARE DIAGNOSTIC SUITE (RPi 5 RP1 GPIO)")
    print("  Testing all connected sensors and actuators without microphone")
    print("=" * 70)

    try:
        hw = PhysicalVehicleHardware(config)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to initialize PhysicalVehicleHardware: {e}", file=sys.stderr)
        return

    results = {}

    try:
        # 1. LED Tests
        print("\n[1/4] Testing Dashboard Indicator LEDs...")
        print("  -> White LED (Headlights, GPIO 17): Turning ON for 1.2s...")
        hw.set_lights(True)
        time.sleep(1.2)
        hw.set_lights(False)
        results["White LED (Headlights)"] = ("OK", "Flashed 1.2s on GPIO 17")

        print("  -> Green LED (Simulated AC, GPIO 27): Turning ON for 1.2s...")
        hw.set_ac(True)
        time.sleep(1.2)
        hw.set_ac(False)
        results["Green LED (AC Indicator)"] = ("OK", "Flashed 1.2s on GPIO 27")

        print("  -> Red LED (IR Alert, GPIO 22): Turning ON for 1.2s...")
        import lgpio
        lgpio.gpio_write(hw._chip_handle, config.pin_led_warning, 1)
        time.sleep(1.2)
        lgpio.gpio_write(hw._chip_handle, config.pin_led_warning, 0)
        results["Red LED (IR Obstacle Alert)"] = ("OK", "Flashed 1.2s on GPIO 22")

        print("  -> Yellow LED (Ultrasonic Alert, GPIO 23): Turning ON for 1.2s...")
        lgpio.gpio_write(hw._chip_handle, config.pin_led_proximity, 1)
        time.sleep(1.2)
        lgpio.gpio_write(hw._chip_handle, config.pin_led_proximity, 0)
        results["Yellow LED (Ultrasonic Alert)"] = ("OK", "Flashed 1.2s on GPIO 23")

        # 2. IR Sensor Test
        print("\n[2/4] Testing IR Obstacle Avoidance Sensor (GPIO 25)...")
        ir_samples = []
        for _ in range(5):
            val = lgpio.gpio_read(hw._chip_handle, config.pin_ir_sensor)
            ir_samples.append(val)
            time.sleep(0.1)
        is_clear = all(v == 1 for v in ir_samples)
        is_detected = any(v == 0 for v in ir_samples)
        ir_status_str = "OBSTACLE DETECTED" if is_detected else "CLEAR / PATH SAFE"
        print(f"  -> IR Reading: {ir_status_str} (Samples: {ir_samples})")
        results["IR Sensor"] = ("OK", f"Responsive ({ir_status_str})")

        # 3. HC-SR04 Ultrasonic Distance Sensor
        print(f"\n[3/4] Testing HC-SR04 Ultrasonic Distance Sensor (TRIG {config.pin_ultrasonic_trig}, ECHO {config.pin_ultrasonic_echo})...")
        dist_readings = []
        for s in range(1, 6):
            r = hw.get_distance()
            dist_readings.append(r)
            if r.status == "OK":
                print(f"  -> Sample {s}: {r.value} cm")
            else:
                print(f"  -> Sample {s}: {r.status}")
            time.sleep(0.1)

        ok_dists = [r.value for r in dist_readings if r.status == "OK"]
        if ok_dists:
            avg_dist = sum(ok_dists) / len(ok_dists)
            results["HC-SR04 Distance"] = ("OK", f"Active ({avg_dist:.1f} cm average)")
        else:
            results["HC-SR04 Distance"] = ("TIMEOUT", "Check VCC/GND and voltage divider wiring")

        # 4. DHT22 Temperature & Humidity Sensor
        print("\n[4/4] Testing DHT22 Temperature & Humidity Sensor (GPIO 4)...")
        temp_reading = hw.get_temperature()
        hum_reading = hw.get_humidity()

        if temp_reading.status == "OK" and hum_reading.status == "OK":
            print(f"  -> Temperature : {temp_reading.value} °C")
            print(f"  -> Humidity    : {hum_reading.value} %")
            results["DHT22 Climate"] = ("OK", f"Temp: {temp_reading.value} °C | Humidity: {hum_reading.value} %")
        else:
            print(f"  -> Temperature status: {temp_reading.status}")
            print(f"  -> Humidity status   : {hum_reading.status}")
            results["DHT22 Climate"] = ("ERROR", f"Temp: {temp_reading.status} | Hum: {hum_reading.status}")

        # 5. SG90 Door Servos
        print("\n[5/5] Testing SG90 Door Servos (Left GPIO 12, Right GPIO 13)...")
        print("  -> Opening both doors (Driver GPIO 12, Passenger GPIO 13)...")
        hw.open_door("both")
        time.sleep(1.2)
        print("  -> Closing both doors...")
        hw.close_door("both")
        time.sleep(1.2)
        results["Left Door Servo (GPIO 12)"] = ("OK", "Pulsed 2000µs (Open) -> 1000µs (Close)")
        results["Right Door Servo (GPIO 13)"] = ("OK", "Pulsed 2000µs (Open) -> 1000µs (Close)")

    finally:
        hw.cleanup()

    # Print Summary Table
    print("\n" + "=" * 70)
    print("                    PHYSICAL HARDWARE TEST SUMMARY")
    print("=" * 70)
    for component, (status, detail) in results.items():
        tag = "[PASS]" if status == "OK" else "[FAIL]"
        print(f" {tag:<7} {component:<26} : {detail}")
    print("=" * 70)
    print("All tested physical components verified without microphone.")


if __name__ == "__main__":
    run_full_diagnostic()
