"""Glue: run the cell, poll the sensors, fill the log.

One function, ``simulate``, is the entry point every script and test uses.  It
is also where the *real-time observation* is assembled — the dict of latest
sensor values that a controller or estimator is allowed to see.  Ground truth
never enters it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from weldloop.config import WeldConfig, default_config
from weldloop.interfaces import SensorSample
from weldloop.sim.cell import GroundTruth, TorchCommand, WeldCell
from weldloop.sim.logger import MasterClockLogger
from weldloop.sim.seam import Seam, make_seam
from weldloop.sim.sensors import SensorSuite

__all__ = ["Observation", "Controller", "RunResult", "simulate"]

#: arbitrary constant that separates the sensor RNG tree from the plant's
_SENSOR_STREAM = 0x5E50



@dataclass(slots=True)
class Observation:
    """Latest value of every sensor channel, with its age.

    This is deliberately the *only* thing a controller sees.  It contains no
    ``truth_`` field, it keeps stale values with their timestamps rather than
    pretending they are fresh, and it never fills in a dropout.
    """

    values: dict[str, float] = field(default_factory=dict)
    stamps: dict[str, float] = field(default_factory=dict)
    t: float = 0.0

    def update(self, samples: list[SensorSample], t: float) -> None:
        self.t = t
        for smp in samples:
            if not smp.valid:
                continue
            for name, value in smp.channels.items():
                self.values[name] = value
                self.stamps[name] = smp.t

    #: machine-side feedback: encoder position, actual speeds, drive tacho.
    #: A real cell has all of these; they are not "sensors" in the suite, but
    #: the controller obviously reads them back, and without them it cannot
    #: know where along the seam it is.
    MACHINE_CHANNELS = (
        "rb_s", "rb_v_travel", "rb_weave_offset", "rb_ctwd", "ps_v_wire", "ps_V_set"
    )

    def update_machine(self, truth) -> None:
        """Write the machine-side feedback channels.

        Takes a ``GroundTruth`` because that is what the simulator has, but
        reads only fields a real robot and a real inverter report.  Nothing
        here is unobservable, and no ``truth_`` field is touched.
        """
        v = self.values
        st = self.stamps
        for name, value in (
            ("rb_s", truth.s),
            ("rb_v_travel", truth.v_travel),
            ("rb_weave_offset", truth.weave_offset),
            ("rb_ctwd", truth.command.ctwd),
            ("ps_v_wire", truth.v_wire),
        ):
            v[name] = float(value)
            st[name] = truth.t

    def get(self, name: str, default: float = float("nan")) -> float:
        return self.values.get(name, default)

    def age(self, name: str) -> float:
        """Seconds since this channel last produced a valid value."""
        if name not in self.stamps:
            return float("inf")
        return self.t - self.stamps[name]


class Controller(Protocol):
    """What Phase 4 plugs in here.

    Only ``update`` is required.  Two optional hooks give a controller access
    to the faster timescales, which is what makes the three-layer split real
    rather than decorative:

    ``on_samples(t, samples)``
        Called every simulation tick with the raw sensor block — the software
        equivalent of a DAQ callback.  The adaptive controller uses it to feed
        its 5 kHz V/I ring buffer.
    ``fast_update(t, obs)``
        Called every ``power_source.inner_dt`` (2 ms) for the electrical inner
        loop, between the 20 ms motion-layer decisions.
    """

    def reset(self, cfg: WeldConfig) -> None: ...

    def update(self, t: float, obs: Observation) -> TorchCommand | None: ...


@dataclass(slots=True)
class RunResult:
    """Everything one simulated weld produced."""

    table: object  # LogTable; typed loosely to avoid an import cycle in docs
    seam: Seam
    cfg: WeldConfig
    n_steps: int
    duration: float


def simulate(
    cfg: WeldConfig | None = None,
    *,
    seed: int | None = None,
    seam: Seam | None = None,
    controller: Controller | None = None,
    include_truth: bool = True,
    raw_history: bool = False,
) -> RunResult:
    """Run one weld end to end and return the master-clock log.

    Parameters
    ----------
    controller:
        Optional; called every ``cfg.control.dt`` seconds with the current
        :class:`Observation`.  Returning ``None`` leaves the command unchanged.
        With no controller the cell runs the fixed baseline command, which is
        what Phase 2 needs.
    include_truth:
        Keep the ``truth_`` columns.  Set ``False`` to produce a table shaped
        exactly like a real capture.
    raw_history:
        Also return the per-step ``GroundTruth`` objects in
        ``RunResult.table.raw``; only used by the plotting scripts, because it
        costs memory.
    """
    cfg = cfg or default_config()
    if seed is not None:
        cfg.sim.seed = int(seed)
    seam = seam if seam is not None else make_seam(cfg.seam, cfg.sim.seed)

    cell = WeldCell(cfg, seam=seam, seed=cfg.sim.seed)
    # Sensor noise gets its own root stream, keyed off the same seed, so that
    # changing the sensor suite cannot perturb the plant trajectory.
    suite = SensorSuite(cfg, seam, np.random.default_rng([cfg.sim.seed, _SENSOR_STREAM]))
    logger = MasterClockLogger(cfg.sensors.f_master, include_truth=include_truth)
    obs = Observation()

    if controller is not None:
        controller.reset(cfg)

    control_period = cfg.control.dt
    inner_period = cfg.power_source.inner_dt
    t_next_control = 0.0
    t_next_inner = 0.0
    n = 0
    raw: list[GroundTruth] = []

    on_samples = getattr(controller, "on_samples", None)
    fast_update = getattr(controller, "fast_update", None)

    while not cell.done:
        gt = cell.step()
        samples = suite.poll(gt.t, gt)
        logger.log_samples(samples)
        logger.log_truth(gt)
        logger.log_V_set(cell.power_source.V_set, gt.t)
        obs.update(samples, gt.t)
        obs.update_machine(gt)

        if on_samples is not None:
            on_samples(gt.t, samples)

        if fast_update is not None and gt.t >= t_next_inner:
            t_next_inner += inner_period
            cmd = fast_update(gt.t, obs)
            if cmd is not None:
                cell.command(cmd)

        if controller is not None and gt.t >= t_next_control:
            t_next_control += control_period
            cmd = controller.update(gt.t, obs)
            if cmd is not None:
                cell.command(cmd)

        if raw_history:
            raw.append(gt)
        n += 1

    table = logger.finish()
    if raw_history:
        table.raw = raw  # type: ignore[attr-defined]
    return RunResult(table=table, seam=seam, cfg=cfg, n_steps=n, duration=cell.t)
