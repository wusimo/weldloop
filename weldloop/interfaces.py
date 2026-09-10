"""Abstract hardware boundaries.

Everything in this file is the seam between the demo and a real welding cell.
Each ABC has exactly one simulated implementation in ``weldloop.sim`` today;
a real-hardware adapter subclasses the same ABC and nothing above it changes.

TODO(real-hw): the three adapters to write are
  * ``FronmarkPowerSource(PowerSourceBase)``  — fieldbus / SDK of the inverter
  * ``AbbRobot(RobotBase)``                   — EGM / RSI style motion streaming
  * ``*Sensor(SensorBase)``                   — one per physical sensor
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SensorSample:
    """One time-stamped sensor reading.

    Attributes
    ----------
    t:
        Timestamp on the **master clock** [s].  Every sensor in the cell
        shares this clock; in real hardware it is the PTP / trigger-line time.
    channels:
        Mapping from column name to value.  Column names are the schema of the
        logged table (see ``sim/logger.py``); ``nan`` means "sampled but
        invalid" while a missing sample means "this sensor did not fire".
    valid:
        False when the sensor fired but the reading must not be used
        (profiler dropout from spatter, RGB blinded by smoke, ...).
    """

    t: float
    channels: dict[str, float]
    valid: bool = True


class SensorBase(ABC):
    """A sensor sampling at its own fixed rate off the master clock."""

    #: column names this sensor contributes to the log
    columns: tuple[str, ...] = ()

    def __init__(self, rate_hz: float, rng: np.random.Generator) -> None:
        self.rate_hz = float(rate_hz)
        self._period = 1.0 / float(rate_hz)
        self._rng = rng
        self._t_next = 0.0

    def due(self, t: float) -> bool:
        """True when the master clock has reached this sensor's next sample."""
        return t + 1e-12 >= self._t_next

    def maybe_sample(self, t: float, truth: Any) -> SensorSample | None:
        """Return a sample if the sensor is due at time ``t``, else ``None``."""
        if not self.due(t):
            return None
        self._t_next += self._period
        return self.sample(t, truth)

    @abstractmethod
    def sample(self, t: float, truth: Any) -> SensorSample:
        """Produce one reading.  ``truth`` is the simulator ground truth.

        TODO(real-hw): a real adapter ignores ``truth`` entirely and reads the
        device instead; the return type is unchanged.
        """

    def reset(self) -> None:
        self._t_next = 0.0


class PowerSourceBase(ABC):
    """The welding inverter: ms-timescale current / wire-feed actuator **and**
    the primary process sensor (V and I at kHz)."""

    @abstractmethod
    def command(self, *, I_set: float, v_wire_set: float, arc_len_set: float) -> None:
        """Push new setpoints (called at the inner-loop rate).

        TODO(real-hw): map onto the machine's job/setpoint registers.
        """

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance the actuator dynamics by ``dt`` seconds."""

    @property
    @abstractmethod
    def measured(self) -> dict[str, float]:
        """Instantaneous machine-side quantities (V, I, wire feed, ...)."""


class RobotBase(ABC):
    """The manipulator carrying the torch: 10-100 ms timescale actuator."""

    @abstractmethod
    def command(
        self, *, v_travel: float, weave_amp: float, torch_angle: float, ctwd: float
    ) -> None:
        """Push new motion setpoints.

        TODO(real-hw): stream to EGM / RSI / a motion-blending buffer.
        """

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance the servo dynamics and integrate arc-length position ``s``."""

    @property
    @abstractmethod
    def state(self) -> dict[str, float]:
        """Current torch kinematic state (``s``, ``v_travel``, weave offset, ...)."""
