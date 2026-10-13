"""Abstract hardware interface contract for vehicle sensors and actuators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class SensorReading:
    """Represents a reading from an environmental or proximity sensor."""
    value: float
    unit: str
    is_simulated: bool
    status: str  # "OK", "ERROR", "TIMEOUT", "DISCONNECTED"


@dataclass
class HardwareResponse:
    """Represents the outcome of an actuator command."""
    success: bool
    message: str
    is_simulated: bool


class AbstractVehicleHardware(ABC):
    """Abstract base class establishing the contract for vehicle hardware."""

    @abstractmethod
    def get_temperature(self) -> SensorReading:
        """Reads ambient cabin temperature in degrees Celsius."""
        pass

    @abstractmethod
    def get_humidity(self) -> SensorReading:
        """Reads ambient cabin relative humidity as a percentage."""
        pass

    @abstractmethod
    def get_distance(self) -> SensorReading:
        """Reads distance to the nearest obstacle in centimeters."""
        pass

    @abstractmethod
    def get_sensor_status(self) -> Dict[str, SensorReading]:
        """Returns the operational status and latest readings of all sensors."""
        pass

    @abstractmethod
    def set_lights(self, enable: bool) -> HardwareResponse:
        """Turns simulated vehicle headlights on or off."""
        pass

    @abstractmethod
    def get_lights_status(self) -> bool:
        """Returns True if vehicle headlights are currently on."""
        pass

    @abstractmethod
    def set_ac(self, enable: bool) -> HardwareResponse:
        """Turns simulated vehicle air conditioning on or off."""
        pass

    @abstractmethod
    def get_ac_status(self) -> bool:
        """Returns True if vehicle air conditioning is currently active."""
        pass

    @abstractmethod
    def set_emergency(self, enable: bool) -> HardwareResponse:
        """Enables or disables vehicle emergency hazard mode (flashes all LEDs)."""
        pass

    @abstractmethod
    def get_emergency_status(self) -> bool:
        """Returns True if vehicle emergency hazard mode is active."""
        pass

    @abstractmethod
    def open_door(self, door: str) -> HardwareResponse:
        """Opens specified door: 'driver', 'passenger', or 'both'."""
        pass

    @abstractmethod
    def close_door(self, door: str) -> HardwareResponse:
        """Closes specified door: 'driver', 'passenger', or 'both'."""
        pass

    @abstractmethod
    def get_door_status(self) -> Dict[str, str]:
        """Returns current status of doors ('open' or 'closed')."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Releases GPIO pins or resets hardware states."""
        pass
