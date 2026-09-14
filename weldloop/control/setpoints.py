"""What the motion layer accepts as its *nominal* operating point.

The motion layer regulates around a nominal current, travel speed and weave,
and inside an acceptance band.  Until now all five came from the config, which
quietly assumed the whole seam is one job.  It is not: the planner's entire
output is the statement "this stretch of seam is a different job from that
one".

This module is deliberately on the *control* side of the boundary, not the
planning side.  The control layer declares what it will accept and what the
limits are; a planner — rule-based, an optimiser, or a language model — may
only fill this struct in, and only through :func:`Nominal.clamped`.  Nothing
upstream ever writes a setpoint that has not been through the clamp.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from weldloop._fastmath import clip
from weldloop.config import WeldConfig

__all__ = ["Nominal", "ClampReport"]


@dataclass(frozen=True, slots=True)
class ClampReport:
    """One setpoint that a planner asked for and did not get."""

    field: str
    asked: float
    got: float
    limit: str

    def describe(self, scale: float = 1.0, unit: str = "") -> str:
        return (
            f"{self.field}: {self.asked * scale:.3g}{unit} -> "
            f"{self.got * scale:.3g}{unit} ({self.limit})"
        )


@dataclass(frozen=True, slots=True)
class Nominal:
    """The operating point the motion layer regulates around."""

    I_set: float          # [A]
    v_travel: float       # [m/s]
    weave_amp: float      # [m]
    p_target: float       # [m]
    p_lo: float           # [m]
    p_hi: float           # [m]
    thickness: float      # [m] plate thickness this stretch of seam is cut from

    #: the thinnest/thickest plate the cell is willing to be *told* it is
    #: welding.  A planner that reads "0.3 mm" off a drawing where the fixture
    #: holds 6 mm plate would otherwise talk the safety monitor into tripping
    #: on every tick, or, worse, out of tripping at all.
    THICKNESS_MIN: ClassVar[float] = 1.0e-3
    THICKNESS_MAX: ClassVar[float] = 50.0e-3

    @classmethod
    def from_config(cls, cfg: WeldConfig) -> "Nominal":
        b, c = cfg.baseline, cfg.control
        return cls(
            I_set=b.I_set,
            v_travel=b.v_travel,
            weave_amp=b.weave_amp,
            p_target=c.p_target,
            p_lo=c.p_lo,
            p_hi=c.p_hi,
            thickness=cfg.joint.thickness,
        )

    # -- the only door into the control layer ----------------------------
    def clamped(self, cfg: WeldConfig) -> tuple["Nominal", tuple[ClampReport, ...]]:
        """Force this operating point inside the machine and procedure limits.

        Returns the admissible point and a record of everything that had to be
        moved.  The record is the interesting half: a planner that has to be
        clamped is a planner that asked for something the cell cannot do, and
        on real hardware that is the event you want in the log.

        TODO(real-hw): on a real cell these limits come from the machine's own
        parameter table and the qualified WPS, not from a config file, and the
        clamp report goes to the procedure-deviation log.
        """
        ctl, rb = cfg.control, cfg.robot
        reports: list[ClampReport] = []

        def _c(name: str, value: float, lo: float, hi: float, limit: str) -> float:
            got = clip(float(value), lo, hi)
            if got != float(value):
                reports.append(ClampReport(name, float(value), got, limit))
            return got

        I = _c("I_set", self.I_set, ctl.I_min_cmd, ctl.I_max_cmd, "motion-layer current range")
        v = _c("v_travel", self.v_travel, rb.v_travel_min, rb.v_travel_max, "robot travel limits")
        a = _c("weave_amp", self.weave_amp, 0.0, rb.weave_amp_max, "robot weave limit")

        # the acceptance band is a procedure decision, so it is clamped against
        # the plate, not against the machine: never deeper than the plate and
        # never a band a controller cannot sit inside.  Which plate is itself
        # something a planner declares, so it gets clamped first.
        h = _c("thickness", self.thickness, self.THICKNESS_MIN, self.THICKNESS_MAX,
               "plausible plate thickness")
        lo = _c("p_lo", self.p_lo, 0.5e-3, ctl.bt_margin * h, "plate thickness")
        hi = _c("p_hi", self.p_hi, 0.5e-3, ctl.bt_margin * h, "plate thickness")
        if hi < lo:
            reports.append(ClampReport("p_hi", hi, lo, "band inverted"))
            hi = lo
        tgt = _c("p_target", self.p_target, lo, hi, "acceptance band")

        return (
            replace(self, I_set=I, v_travel=v, weave_amp=a, p_target=tgt, p_lo=lo,
                    p_hi=hi, thickness=h),
            tuple(reports),
        )
