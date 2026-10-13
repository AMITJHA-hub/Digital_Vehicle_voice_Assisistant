"""Hardware abstraction layer for simulated and physical vehicles."""

from .interface import (
    AbstractVehicleHardware,
    SensorReading,
    HardwareResponse,
)
from .simulation import SimulatedVehicleHardware

__all__ = [
    "AbstractVehicleHardware",
    "SensorReading",
    "HardwareResponse",
    "SimulatedVehicleHardware",
    "get_hardware",
]


def get_hardware(mode: str = "simulation") -> AbstractVehicleHardware:
    """Returns the requested hardware implementation.

    Defaults safely to simulation mode.
    Physical mode will only be instantiated when explicitly configured.
    """
    mode = (mode or "simulation").strip().lower()
    if mode == "physical":
        from .physical import PhysicalVehicleHardware
        return PhysicalVehicleHardware()
    return SimulatedVehicleHardware()
