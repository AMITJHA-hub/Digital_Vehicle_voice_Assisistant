"""Simulated vehicle hardware implementation for safe testing without GPIO."""

from typing import Dict
from .interface import (
    AbstractVehicleHardware,
    SensorReading,
    HardwareResponse,
)


class SimulatedVehicleHardware(AbstractVehicleHardware):
    """Simulates vehicle sensors and actuators completely in memory."""

    def __init__(self):
        self.lights_on: bool = False
        self.ac_on: bool = False
        self.emergency_mode: bool = False
        self.doors: Dict[str, str] = {
            "driver": "closed",
            "passenger": "closed",
        }
        # Simulated sensor telemetry
        self._simulated_temp: float = 23.5
        self._simulated_humidity: float = 48.0
        self._simulated_distance: float = 85.0

    def get_temperature(self) -> SensorReading:
        return SensorReading(
            value=self._simulated_temp,
            unit="°C",
            is_simulated=True,
            status="OK",
        )

    def get_humidity(self) -> SensorReading:
        return SensorReading(
            value=self._simulated_humidity,
            unit="%",
            is_simulated=True,
            status="OK",
        )

    def get_distance(self) -> SensorReading:
        return SensorReading(
            value=self._simulated_distance,
            unit="cm",
            is_simulated=True,
            status="OK",
        )

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
            message=f"Simulated vehicle headlights switched {state_str}.",
            is_simulated=True,
        )

    def get_lights_status(self) -> bool:
        return self.lights_on

    def set_ac(self, enable: bool) -> HardwareResponse:
        self.ac_on = enable
        state_str = "ON" if enable else "OFF"
        return HardwareResponse(
            success=True,
            message=f"Simulated AC switched {state_str}.",
            is_simulated=True,
        )

    def get_ac_status(self) -> bool:
        return self.ac_on

    def set_emergency(self, enable: bool) -> HardwareResponse:
        self.emergency_mode = enable
        state_str = "ACTIVATED" if enable else "DEACTIVATED"
        return HardwareResponse(
            success=True,
            message=f"Simulated vehicle emergency hazard mode {state_str}.",
            is_simulated=True,
        )

    def get_emergency_status(self) -> bool:
        return self.emergency_mode

    def open_door(self, door: str) -> HardwareResponse:
        target = door.lower()
        if target in ("driver", "passenger"):
            self.doors[target] = "open"
            return HardwareResponse(
                success=True,
                message=f"Simulated {target} door opened.",
                is_simulated=True,
            )
        elif target == "both":
            self.doors["driver"] = "open"
            self.doors["passenger"] = "open"
            return HardwareResponse(
                success=True,
                message="Simulated both driver and passenger doors opened.",
                is_simulated=True,
            )
        else:
            return HardwareResponse(
                success=False,
                message=f"Invalid door target '{door}'.",
                is_simulated=True,
            )

    def close_door(self, door: str) -> HardwareResponse:
        target = door.lower()
        if target in ("driver", "passenger"):
            self.doors[target] = "closed"
            return HardwareResponse(
                success=True,
                message=f"Simulated {target} door closed.",
                is_simulated=True,
            )
        elif target == "both":
            self.doors["driver"] = "closed"
            self.doors["passenger"] = "closed"
            return HardwareResponse(
                success=True,
                message="Simulated both driver and passenger doors closed.",
                is_simulated=True,
            )
        else:
            return HardwareResponse(
                success=False,
                message=f"Invalid door target '{door}'.",
                is_simulated=True,
            )

    def get_door_status(self) -> Dict[str, str]:
        return dict(self.doors)

    def cleanup(self) -> None:
        """No hardware cleanup required in simulation."""
        pass
