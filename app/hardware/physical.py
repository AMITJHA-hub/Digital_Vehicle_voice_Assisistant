"""Physical hardware driver implementation for Raspberry Pi 5 (RP1 GPIO)."""

import logging
import threading
import time
from typing import Dict, Optional, Tuple

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False

from ..config import config, AppConfig
from .interface import (
    AbstractVehicleHardware,
    SensorReading,
    HardwareResponse,
)

logger = logging.getLogger(__name__)


class PhysicalVehicleHardware(AbstractVehicleHardware):
    """Controls physical breadboard components connected to the Raspberry Pi 5 RP1 chip.

    Active features:
    - Background dynamic monitoring loop for real-time proximity safety and AC blinking.
    - HC-SR04 distance rules:
        < 5 cm: Red Warning LED solid ON
        5 to 10 cm: Red Warning LED rapid blink alert
        > 10 cm: Red Warning LED OFF
    - IR sensor: Proximity detection keeps Red Warning LED ON when obstacle is detected.
    - AC status: Green LED blinks continuously while AC is ON; turns OFF when AC is OFF.
    - Emergency Mode: All 3 LEDs (White, Green, Red) flash simultaneously in hazard pattern.
    - DHT22: Single-wire pulse decoding for cabin climate.
    - Servos: Brownout lockout protection.
    """

    def __init__(self, app_config: Optional[AppConfig] = None):
        self.config = app_config or config
        self._chip_handle: Optional[int] = None
        self.lights_on: bool = False
        self.ac_on: bool = False
        self.emergency_mode: bool = False
        self.latest_distance: float = 999.0
        self.ir_detected: bool = False
        self.doors: Dict[str, str] = {
            "driver": "closed",
            "passenger": "closed",
        }
        self.servos_enabled: bool = True  # Unlocked with external 5V power supply

        self._last_dht_read_time: float = 0.0
        self._cached_temp: Optional[float] = None
        self._cached_humidity: Optional[float] = None

        self._gpio_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        if not LGPIO_AVAILABLE:
            raise RuntimeError(
                "The 'lgpio' library is not available. Ensure python3-lgpio is installed."
            )

        self._init_gpio()
        self._start_monitor()

    def _init_gpio(self) -> None:
        """Initializes the RP1 GPIO chip (gpiochip 0) and sets pin modes."""
        try:
            self._chip_handle = lgpio.gpiochip_open(0)
            logger.info("Opened RP1 gpiochip 0 successfully.")

            # Configure LED Outputs (Low by default)
            lgpio.gpio_claim_output(self._chip_handle, self.config.pin_led_headlights, 0)
            lgpio.gpio_claim_output(self._chip_handle, self.config.pin_led_ac, 0)
            lgpio.gpio_claim_output(self._chip_handle, self.config.pin_led_warning, 0)
            lgpio.gpio_claim_output(self._chip_handle, self.config.pin_led_proximity, 0)

            # Configure Ultrasonic Sensor
            lgpio.gpio_claim_output(self._chip_handle, self.config.pin_ultrasonic_trig, 0)
            lgpio.gpio_claim_input(self._chip_handle, self.config.pin_ultrasonic_echo)

            # Configure IR obstacle sensor (Digital input with internal pull-up)
            lgpio.gpio_claim_input(self._chip_handle, self.config.pin_ir_sensor)

            # Configure Door Servos (GPIO 12 = Driver/Left Door, GPIO 13 = Passenger/Right Door)
            lgpio.gpio_claim_output(self._chip_handle, self.config.pin_servo_driver_door, 0)
            lgpio.gpio_claim_output(self._chip_handle, self.config.pin_servo_pass_door, 0)

        except Exception as e:
            logger.error(f"Failed to initialize RP1 GPIO pins: {e}")
            self.cleanup()
            raise

    def _start_monitor(self) -> None:
        """Launches the real-time background vehicle safety and actuator thread."""
        self._monitor_thread = threading.Thread(
            target=self._hardware_monitor_loop,
            daemon=True,
            name="VehicleHardwareMonitor",
        )
        self._monitor_thread.start()

    def _hardware_monitor_loop(self) -> None:
        """Continuous real-time control loop for proximity safety and actuator blinking."""
        hazard_state = False
        ac_blink_state = False
        prox_blink_state = False

        last_hazard_toggle = time.time()
        last_ac_toggle = time.time()
        last_prox_toggle = time.time()
        last_sensor_scan = 0.0

        while not self._monitor_stop.is_set():
            now = time.time()

            # 1. Background sensor scanning (~ every 70 ms)
            if now - last_sensor_scan > 0.07:
                last_sensor_scan = now
                with self._gpio_lock:
                    if self._chip_handle is not None:
                        # Poll IR obstacle sensor
                        try:
                            ir_val = lgpio.gpio_read(self._chip_handle, self.config.pin_ir_sensor)
                            self.ir_detected = (ir_val == 0)
                        except Exception:
                            pass

                        # Measure HC-SR04 distance
                        d = self._measure_ultrasonic_distance()
                        if d is not None:
                            self.latest_distance = d

            # 2. Dynamic LED actuation
            with self._gpio_lock:
                if self._chip_handle is not None:
                    if self.emergency_mode:
                        # Emergency Mode: All 4 LEDs flash simultaneously in hazard pattern
                        if now - last_hazard_toggle > 0.30:
                            hazard_state = not hazard_state
                            last_hazard_toggle = now
                        val = 1 if hazard_state else 0
                        lgpio.gpio_write(self._chip_handle, self.config.pin_led_headlights, val)
                        lgpio.gpio_write(self._chip_handle, self.config.pin_led_ac, val)
                        lgpio.gpio_write(self._chip_handle, self.config.pin_led_warning, val)
                        lgpio.gpio_write(self._chip_handle, self.config.pin_led_proximity, val)

                    else:
                        # A. White LED: Headlights (solid ON if lights_on else OFF)
                        lgpio.gpio_write(
                            self._chip_handle,
                            self.config.pin_led_headlights,
                            1 if self.lights_on else 0,
                        )

                        # B. Green LED: AC (blinks while AC is ON, OFF when AC is OFF)
                        if self.ac_on:
                            if now - last_ac_toggle > 0.35:
                                ac_blink_state = not ac_blink_state
                                last_ac_toggle = now
                            lgpio.gpio_write(
                                self._chip_handle,
                                self.config.pin_led_ac,
                                1 if ac_blink_state else 0,
                            )
                        else:
                            ac_blink_state = False
                            lgpio.gpio_write(self._chip_handle, self.config.pin_led_ac, 0)

                        # C. Red LED: IR Obstacle Sensor ONLY
                        # Solid ON when obstacle is detected near (~4 cm), else OFF
                        lgpio.gpio_write(
                            self._chip_handle,
                            self.config.pin_led_warning,
                            1 if self.ir_detected else 0,
                        )

                        # D. Yellow LED: Front Ultrasonic Sensor (HC-SR04) ONLY
                        # < 5 cm -> CONSTANT ON
                        # 5 to 10 cm -> RAPID BLINKING
                        # > 10 cm -> OFF
                        if self.latest_distance < 5.0:
                            lgpio.gpio_write(self._chip_handle, self.config.pin_led_proximity, 1)
                        elif 5.0 <= self.latest_distance <= 10.0:
                            if now - last_prox_toggle > 0.15:
                                prox_blink_state = not prox_blink_state
                                last_prox_toggle = now
                            lgpio.gpio_write(
                                self._chip_handle,
                                self.config.pin_led_proximity,
                                1 if prox_blink_state else 0,
                            )
                        else:
                            prox_blink_state = False
                            lgpio.gpio_write(self._chip_handle, self.config.pin_led_proximity, 0)

            time.sleep(0.025)

    def _measure_ultrasonic_distance(self) -> Optional[float]:
        """Low-level HC-SR04 pulse measurement."""
        trig = self.config.pin_ultrasonic_trig
        echo = self.config.pin_ultrasonic_echo

        try:
            lgpio.gpio_write(self._chip_handle, trig, 0)
            time.sleep(0.001)

            # 10us trigger pulse
            lgpio.gpio_write(self._chip_handle, trig, 1)
            time.sleep(0.000010)
            lgpio.gpio_write(self._chip_handle, trig, 0)

            t0 = time.perf_counter()
            timeout = t0 + 0.030

            pulse_start = time.perf_counter()
            while lgpio.gpio_read(self._chip_handle, echo) == 0:
                pulse_start = time.perf_counter()
                if pulse_start > timeout:
                    return None

            pulse_end = pulse_start
            while lgpio.gpio_read(self._chip_handle, echo) == 1:
                pulse_end = time.perf_counter()
                if pulse_end > timeout:
                    return None

            duration = pulse_end - pulse_start
            distance = (duration * 34300.0) / 2.0
            if 2.0 <= distance <= 400.0:
                return round(distance, 1)
            return None
        except Exception:
            return None

    def get_distance(self) -> SensorReading:
        """Reads distance using the HC-SR04 ultrasonic sensor with safety timeouts."""
        with self._gpio_lock:
            d = self._measure_ultrasonic_distance()

        if d is not None:
            self.latest_distance = d
            return SensorReading(value=d, unit="cm", is_simulated=False, status="OK")
        elif self.latest_distance < 400.0:
            return SensorReading(
                value=self.latest_distance,
                unit="cm",
                is_simulated=False,
                status="OK",
            )
        else:
            return SensorReading(value=0.0, unit="cm", is_simulated=False, status="TIMEOUT")

    def _read_dht22(self) -> Tuple[Optional[float], Optional[float]]:
        """Reads temperature and humidity from DHT22 on GPIO 4 with retry and cache."""
        now = time.time()
        if self._cached_temp is not None and (now - self._last_dht_read_time) < 2.0:
            return self._cached_temp, self._cached_humidity

        pin = self.config.pin_dht22
        for _ in range(3):
            try:
                with self._gpio_lock:
                    lgpio.gpio_claim_output(self._chip_handle, pin, 1)
                    time.sleep(0.02)
                    lgpio.gpio_write(self._chip_handle, pin, 0)
                    time.sleep(0.018)
                    lgpio.gpio_write(self._chip_handle, pin, 1)
                    time.sleep(0.000030)

                    lgpio.gpio_free(self._chip_handle, pin)
                    lgpio.gpio_claim_input(self._chip_handle, pin, lgpio.SET_PULL_UP)

                    t_start = time.perf_counter()
                    pulses = []
                    last_val = lgpio.gpio_read(self._chip_handle, pin)
                    last_t = t_start

                    while (time.perf_counter() - t_start) < 0.010:
                        v = lgpio.gpio_read(self._chip_handle, pin)
                        t_now = time.perf_counter()
                        if v != last_val:
                            pulses.append((t_now - last_t, last_val))
                            last_t = t_now
                            last_val = v

                high_durations = [dur for dur, val in pulses if val == 1]
                if len(high_durations) >= 41:
                    bit_highs = high_durations[1:41]
                elif len(high_durations) == 40:
                    bit_highs = high_durations
                else:
                    time.sleep(0.2)
                    continue

                bits = [1 if d > 0.000045 else 0 for d in bit_highs]
                bytes_data = []
                for i in range(0, 40, 8):
                    byte_val = 0
                    for b in bits[i:i+8]:
                        byte_val = (byte_val << 1) | b
                    bytes_data.append(byte_val)

                h_high, h_low, t_high, t_low, checksum = bytes_data
                calc_check = (h_high + h_low + t_high + t_low) & 0xFF
                if calc_check != checksum:
                    time.sleep(0.2)
                    continue

                hum = ((h_high << 8) | h_low) / 10.0
                t_val = (((t_high & 0x7F) << 8) | t_low) / 10.0
                if t_high & 0x80:
                    t_val = -t_val

                self._cached_temp = round(t_val, 1)
                self._cached_humidity = round(hum, 1)
                self._last_dht_read_time = time.time()
                return self._cached_temp, self._cached_humidity

            except Exception as e:
                logger.debug(f"DHT22 read attempt failed: {e}")
                time.sleep(0.2)

        return self._cached_temp, self._cached_humidity

    def get_temperature(self) -> SensorReading:
        temp, _ = self._read_dht22()
        if temp is not None:
            return SensorReading(value=temp, unit="°C", is_simulated=False, status="OK")
        return SensorReading(value=0.0, unit="°C", is_simulated=False, status="ERROR")

    def get_humidity(self) -> SensorReading:
        _, hum = self._read_dht22()
        if hum is not None:
            return SensorReading(value=hum, unit="%", is_simulated=False, status="OK")
        return SensorReading(value=0.0, unit="%", is_simulated=False, status="ERROR")

    def get_sensor_status(self) -> Dict[str, SensorReading]:
        return {
            "temperature": self.get_temperature(),
            "humidity": self.get_humidity(),
            "distance": self.get_distance(),
        }

    def set_lights(self, enable: bool) -> HardwareResponse:
        self.lights_on = enable
        state_str = "ON" if enable else "OFF"
        return HardwareResponse(
            success=True,
            message=f"Vehicle headlights turned {state_str}.",
            is_simulated=False,
        )

    def get_lights_status(self) -> bool:
        return self.lights_on

    def set_ac(self, enable: bool) -> HardwareResponse:
        self.ac_on = enable
        state_str = "ON" if enable else "OFF"
        action_note = " (blinking active)" if enable else ""
        return HardwareResponse(
            success=True,
            message=f"Vehicle air conditioning turned {state_str}{action_note}.",
            is_simulated=False,
        )

    def get_ac_status(self) -> bool:
        return self.ac_on

    def set_emergency(self, enable: bool) -> HardwareResponse:
        self.emergency_mode = enable
        state_str = "ACTIVATED" if enable else "DEACTIVATED"
        return HardwareResponse(
            success=True,
            message=f"Vehicle emergency hazard mode {state_str}.",
            is_simulated=False,
        )

    def get_emergency_status(self) -> bool:
        return self.emergency_mode

    def _actuate_servo(self, pin: int, pulse_us: int) -> None:
        """Pushes pulse to servo, allows physical movement, then releases holding torque."""
        def _worker():
            try:
                with self._gpio_lock:
                    if self._chip_handle is not None:
                        lgpio.tx_servo(self._chip_handle, pin, pulse_us)
                time.sleep(0.6)
                with self._gpio_lock:
                    if self._chip_handle is not None:
                        lgpio.tx_servo(self._chip_handle, pin, 0)
            except Exception as e:
                logger.warning(f"Error actuating servo on pin {pin}: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def open_door(self, door: str) -> HardwareResponse:
        target = door.lower()
        if not self.servos_enabled:
            return HardwareResponse(
                success=False,
                message="Physical servo movement is locked for electrical safety. External power supply required.",
                is_simulated=False,
            )

        # Pulse width 2000us = Open position (~90-180 deg)
        if target in ("driver", "left"):
            self.doors["driver"] = "open"
            self._actuate_servo(self.config.pin_servo_driver_door, 2000)
            target_str = "left (driver)"
        elif target in ("passenger", "right"):
            self.doors["passenger"] = "open"
            self._actuate_servo(self.config.pin_servo_pass_door, 2000)
            target_str = "right (passenger)"
        elif target in ("both", "all"):
            self.doors["driver"] = "open"
            self.doors["passenger"] = "open"
            self._actuate_servo(self.config.pin_servo_driver_door, 2000)
            self._actuate_servo(self.config.pin_servo_pass_door, 2000)
            target_str = "both"
        else:
            return HardwareResponse(False, f"Unknown door target: '{door}'.", is_simulated=False)

        return HardwareResponse(True, f"Physical {target_str} door opened.", is_simulated=False)

    def close_door(self, door: str) -> HardwareResponse:
        target = door.lower()
        if not self.servos_enabled:
            return HardwareResponse(
                success=False,
                message="Physical servo movement is locked for electrical safety. External power supply required.",
                is_simulated=False,
            )

        # Pulse width 1000us = Closed position (0 deg)
        if target in ("driver", "left"):
            self.doors["driver"] = "closed"
            self._actuate_servo(self.config.pin_servo_driver_door, 1000)
            target_str = "left (driver)"
        elif target in ("passenger", "right"):
            self.doors["passenger"] = "closed"
            self._actuate_servo(self.config.pin_servo_pass_door, 1000)
            target_str = "right (passenger)"
        elif target in ("both", "all"):
            self.doors["driver"] = "closed"
            self.doors["passenger"] = "closed"
            self._actuate_servo(self.config.pin_servo_driver_door, 1000)
            self._actuate_servo(self.config.pin_servo_pass_door, 1000)
            target_str = "both"
        else:
            return HardwareResponse(False, f"Unknown door target: '{door}'.", is_simulated=False)

        return HardwareResponse(True, f"Physical {target_str} door closed.", is_simulated=False)

    def get_door_status(self) -> Dict[str, str]:
        return dict(self.doors)

    def cleanup(self) -> None:
        """Safely stops background threads and releases RP1 GPIO handles."""
        self._monitor_stop.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)

        with self._gpio_lock:
            if self._chip_handle is not None:
                try:
                    lgpio.gpio_write(self._chip_handle, self.config.pin_led_headlights, 0)
                    lgpio.gpio_write(self._chip_handle, self.config.pin_led_ac, 0)
                    lgpio.gpio_write(self._chip_handle, self.config.pin_led_warning, 0)
                    lgpio.gpio_write(self._chip_handle, self.config.pin_led_proximity, 0)
                    try:
                        lgpio.tx_servo(self._chip_handle, self.config.pin_servo_driver_door, 0)
                        lgpio.tx_servo(self._chip_handle, self.config.pin_servo_pass_door, 0)
                    except Exception:
                        pass
                    lgpio.gpiochip_close(self._chip_handle)
                    logger.info("Closed RP1 gpiochip 0 cleanly.")
                except Exception as e:
                    logger.warning(f"Error during GPIO cleanup: {e}")
                finally:
                    self._chip_handle = None
