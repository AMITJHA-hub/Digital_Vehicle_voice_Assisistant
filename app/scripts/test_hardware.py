"""Hardware diagnostic and interactive breadboard testing tool for Raspberry Pi 5."""

import sys
import time

try:
    import lgpio
except ImportError:
    print("Error: lgpio library not found. Run in the virtual environment.", file=sys.stderr)
    sys.exit(1)

from app.config import config


def print_wiring_guide():
    """Prints complete electrical wiring specifications and safety warnings."""
    print("""
================================================================================
               RASPBERRY PI 5 - COCKPIT DEMO WIRING MAP & SAFETY GUIDE
================================================================================

  MANDATORY ELECTRICAL SAFETY RULES:
  1. NEVER connect a 5V signal directly to a Raspberry Pi GPIO pin (Max: 3.3V).
  2. The HC-SR04 ECHO pin outputs 5.0V. YOU MUST USE A VOLTAGE DIVIDER:
       - HC-SR04 ECHO Pin -> 1 kΩ Resistor (R1) -> Node A
       - Node A -> Pi GPIO 24 (Physical Pin 18)
       - Node A -> 1 kΩ + 330 Ω Resistors (R2) -> Pi GND
       This divides 5.0V down to ~2.85V, safely protecting your Pi 5 RP1 chip!
  3. All LEDs require a 330 Ω current-limiting resistor in series with the anode.
  4. DO NOT power SG90 servo motors from Pi 5V header during active testing.
     Servos require an external dedicated 5V power supply with a shared GND.

--------------------------------------------------------------------------------
 COMPONENT PIN CONNECTIONS:
--------------------------------------------------------------------------------
 [1] White LED (Simulated Headlights):
     - Anode (+) -> 330 Ω Resistor -> GPIO 17 (Physical Pin 11)
     - Cathode (-) -> GND (Physical Pin 9 or Pin 6)

 [2] Green LED (Simulated AC Active Indicator):
     - Anode (+) -> 330 Ω Resistor -> GPIO 27 (Physical Pin 13)
     - Cathode (-) -> GND (Physical Pin 14)

 [3] Red LED (IR Obstacle Sensor Warning):
     - Anode (+) -> 330 Ω Resistor -> GPIO 22 (Physical Pin 15)
     - Cathode (-) -> GND (Physical Pin 14 or breadboard GND rail)

 [4] Yellow LED (Ultrasonic Proximity Alert):
     - Anode (+) -> 330 Ω Resistor -> GPIO 23 (Physical Pin 16)
     - Cathode (-) -> GND (Physical Pin 14 or breadboard GND rail)

 [5] HC-SR04 Ultrasonic Distance Sensor (Re-routed):
     - VCC  -> 5V Power (Physical Pin 2 or Pin 4)
     - GND  -> GND (Physical Pin 30 or breadboard GND rail)
     - TRIG -> GPIO 5 (Physical Pin 29)
     - ECHO -> Through Voltage Divider (see Rule 2) -> GPIO 6 (Physical Pin 31)

 [6] IR Obstacle Sensor:
     - VCC  -> 3.3V Power (Physical Pin 1)
     - GND  -> GND (Physical Pin 9 or breadboard GND rail)
     - OUT  -> GPIO 25 (Physical Pin 22)

 [7] DHT22 / AM2302 (Temperature & Humidity):
     - VCC  -> 3.3V Power (Physical Pin 17)
     - GND  -> GND (Physical Pin 25 or breadboard GND rail)
     - DATA -> GPIO 4 (Physical Pin 7) + 1 kΩ Pull-up to 3.3V

 [8] SG90 Servo Motors (Left & Right Doors):
     - Left (Driver) Door Signal  -> GPIO 12 (Physical Pin 32)
     - Right (Passenger) Door Signal -> GPIO 13 (Physical Pin 33)
     - VCC (Red wire) -> External 5V Power Supply (+)
     - GND (Brown/Black wire) -> External 5V Power Supply (-)
     - MANDATORY COMMON GND: External Supply (-) -> Pi GND (Physical Pin 34)
================================================================================
""")


def test_led(chip, pin, name, duration=3):
    print(f"\n[Testing {name}] Blinking GPIO {pin} for {duration} seconds...")
    try:
        lgpio.gpio_claim_output(chip, pin, 0)
        for _ in range(duration):
            lgpio.gpio_write(chip, pin, 1)
            print(f"  -> {name} ON (HIGH)")
            time.sleep(0.5)
            lgpio.gpio_write(chip, pin, 0)
            print(f"  -> {name} OFF (LOW)")
            time.sleep(0.5)
        print(f"  [PASS] {name} test complete.")
    except Exception as e:
        print(f"  [ERROR] {e}")


def test_ultrasonic(chip, trig, echo, samples=5):
    print(f"\n[Testing Ultrasonic HC-SR04] Measuring {samples} readings on TRIG {trig}, ECHO {echo}...")
    try:
        lgpio.gpio_claim_output(chip, trig, 0)
        lgpio.gpio_claim_input(chip, echo)

        for i in range(1, samples + 1):
            lgpio.gpio_write(chip, trig, 0)
            time.sleep(0.01)

            # 10us trigger pulse
            lgpio.gpio_write(chip, trig, 1)
            time.sleep(0.00001)
            lgpio.gpio_write(chip, trig, 0)

            t0 = time.perf_counter()
            start = t0
            while lgpio.gpio_read(chip, echo) == 0:
                start = time.perf_counter()
                if start - t0 > 0.05:
                    break
            else:
                t1 = time.perf_counter()
                end = t1
                while lgpio.gpio_read(chip, echo) == 1:
                    end = time.perf_counter()
                    if end - t1 > 0.05:
                        break
                else:
                    dist = ((end - start) * 34300.0) / 2.0
                    print(f"  Sample {i}: {dist:.1f} cm")
                    time.sleep(0.2)
                    continue

            print(f"  Sample {i}: [TIMEOUT] No echo pulse received.")
            time.sleep(0.2)

    except Exception as e:
        print(f"  [ERROR] {e}")


def test_ir_sensor(chip, pin, duration=5):
    print(f"\n[Testing IR Sensor] Reading GPIO {pin} for {duration} seconds (Wave hand in front)...")
    try:
        lgpio.gpio_claim_input(chip, pin)
        end_time = time.time() + duration
        while time.time() < end_time:
            val = lgpio.gpio_read(chip, pin)
            status = "OBSTACLE DETECTED (LOW)" if val == 0 else "CLEAR (HIGH)"
            print(f"\r  IR Status: {status}   ", end="", flush=True)
            time.sleep(0.1)
        print("\n  [PASS] IR test finished.")
    except Exception as e:
        print(f"\n  [ERROR] {e}")


def test_servo(chip, pin, name):
    print(f"\n[Testing {name}] Moving GPIO {pin} (50 Hz PWM)...")
    try:
        lgpio.gpio_claim_output(chip, pin, 0)
        print("  -> Sweeping to OPEN position (2000 µs)...")
        lgpio.tx_servo(chip, pin, 2000)
        time.sleep(1.0)
        print("  -> Sweeping to CLOSED position (1000 µs)...")
        lgpio.tx_servo(chip, pin, 1000)
        time.sleep(1.0)
        lgpio.tx_servo(chip, pin, 0)
        print(f"  [PASS] {name} test complete.")
    except Exception as e:
        print(f"  [ERROR] {e}")


def main():
    print_wiring_guide()

    print("Opening RP1 gpiochip 0...")
    try:
        chip = lgpio.gpiochip_open(0)
    except Exception as e:
        print(f"Error opening gpiochip 0: {e}", file=sys.stderr)
        return

    try:
        while True:
            print("\nSelect a test to run:")
            print("  1. Blink Headlights White LED (GPIO 17)")
            print("  2. Blink AC Green LED (GPIO 27)")
            print("  3. Blink IR Warning Red LED (GPIO 22)")
            print("  4. Blink Ultrasonic Warning Yellow LED (GPIO 23)")
            print(f"  5. Read HC-SR04 Ultrasonic Distance (TRIG {config.pin_ultrasonic_trig}, ECHO {config.pin_ultrasonic_echo})")
            print("  6. Read IR Obstacle Sensor (GPIO 25)")
            print("  7. Test Left (Driver) Door Servo (GPIO 12)")
            print("  8. Test Right (Passenger) Door Servo (GPIO 13)")
            print("  9. Print Wiring Diagram again")
            print("  10. Exit")

            choice = input("\nEnter choice (1-10): ").strip()
            if choice == "1":
                test_led(chip, config.pin_led_headlights, "White Headlights LED")
            elif choice == "2":
                test_led(chip, config.pin_led_ac, "Green AC LED")
            elif choice == "3":
                test_led(chip, config.pin_led_warning, "Red IR Warning LED")
            elif choice == "4":
                test_led(chip, config.pin_led_proximity, "Yellow Ultrasonic Warning LED")
            elif choice == "5":
                test_ultrasonic(chip, config.pin_ultrasonic_trig, config.pin_ultrasonic_echo)
            elif choice == "6":
                test_ir_sensor(chip, config.pin_ir_sensor)
            elif choice == "7":
                test_servo(chip, config.pin_servo_driver_door, "Left (Driver) Door Servo")
            elif choice == "8":
                test_servo(chip, config.pin_servo_pass_door, "Right (Passenger) Door Servo")
            elif choice == "9":
                print_wiring_guide()
            elif choice in ("10", "q", "exit"):
                break
            else:
                print("Invalid choice.")
    finally:
        lgpio.gpiochip_close(chip)
        print("GPIO resources closed cleanly.")


if __name__ == "__main__":
    main()
