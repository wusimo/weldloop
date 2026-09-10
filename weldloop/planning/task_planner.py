"""Task planning: seam description -> segments + initial process window.

**This layer is deliberately not in the real-time loop, and the reason is
timescales, not taste.**

    power-source inner loop   ~1 ms      the inverter's own regulation
    torch-motion layer        20 ms      control/motion_layer.py
    task planning             s - min    this file

Penetration responds to a gap change with a time constant of order 100 ms and
burn-through develops in under a second (both measured in Phase 1).  Anything
that decides at seconds-to-minutes cannot close that loop, and anything that
could close it has to run at 50 Hz on the cell controller with a bounded
worst-case latency.

TODO(llm-planner): the natural hook is :meth:`TaskPlanner.plan`.  A language
model — or any offline optimiser — can reasonably be asked to turn a job
description, a WPS, a drawing or a fit-up scan into the segment list and the
starting process window returned here.  What it must **not** be asked to do:

* stream torch commands during the arc.  It is orders of magnitude too slow,
  and its natural input (a camera) is unusable for ~74 % of frames while the
  arc is lit — see the RGB measurements in Phase 2 and the RGB control arm in
  Phase 4, which pays 62 % extra cycle time for exactly this reason.
* replace the state estimator.  The pool state is not directly observable and
  the controller consumes a covariance; a point prediction with no calibrated
  uncertainty is strictly worse than the EKF for that job.

A planner's output is a *starting point*.  Everything downstream is expected
to move away from it as the joint turns out to be different from the drawing —
which is the whole point of the layer below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from weldloop.config import WeldConfig, default_config
from weldloop.physics.arc import melting_rate
from weldloop.sim.seam import Seam

__all__ = ["SeamDescription", "Segment", "WeldPlan", "TaskPlanner"]


@dataclass(frozen=True, slots=True)
class SeamDescription:
    """What a planner is given: the job, not the process."""

    length: float                     # [m]
    thickness: float                  # [m]
    joint_type: str = "square_butt"
    material: str = "mild_steel"
    position: str = "PA"              # flat
    gap_nominal: float = 0.0          # [m], from the drawing
    gap_tolerance: float = 4.0e-3     # [m], the fit-up the cell must survive
    notes: str = ""


@dataclass(frozen=True, slots=True)
class Segment:
    """One stretch of seam with its own starting parameters."""

    s_start: float
    s_end: float
    gap_class: str                    # "tight" | "nominal" | "wide"
    I_set: float
    v_travel: float
    v_wire: float
    weave_amp: float
    rationale: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WeldPlan:
    """A plan: segments, the process window the motion layer must respect, and
    an honest statement of what the plan does not know."""

    segments: tuple[Segment, ...]
    p_target: float
    p_lo: float
    p_hi: float
    estimated_cycle_time: float
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "segments": [s.as_dict() for s in self.segments],
            "p_target": self.p_target,
            "p_lo": self.p_lo,
            "p_hi": self.p_hi,
            "estimated_cycle_time": self.estimated_cycle_time,
            "assumptions": list(self.assumptions),
        }


class TaskPlanner:
    """A deliberately dumb rule-based planner.

    It exists to define the interface and to produce a sane starting point, not
    to be clever.  Replacing the body of :meth:`plan` with a call to a language
    model changes nothing downstream — which is the point of writing it as a
    seam in the first place.
    """

    def __init__(self, cfg: WeldConfig | None = None) -> None:
        self.cfg = cfg or default_config()

    # -- the hook --------------------------------------------------------
    def plan(
        self, seam: SeamDescription, measured: Seam | None = None, n_segments: int = 3
    ) -> WeldPlan:
        """Produce a segment list and the initial process window.

        Parameters
        ----------
        seam:
            The job as described on paper.
        measured:
            Optionally, a fit-up scan.  When present the segments are cut at
            the gap classes actually measured rather than assumed; this is the
            realistic phase-1 workflow (scan the joint cold, then weld it).

        TODO(llm-planner): swap this body for a model call.  Keep the return
        type; keep the ``assumptions`` field honest.
        """
        c = self.cfg
        stickout = c.robot.ctwd_nom - c.arc.L_arc_ref
        edges = np.linspace(0.0, seam.length, n_segments + 1)

        segments = []
        for i in range(n_segments):
            s0, s1 = float(edges[i]), float(edges[i + 1])
            if measured is not None:
                m = (measured.s >= s0) & (measured.s <= s1)
                gap = float(np.percentile(measured.gap[m], 85)) if m.any() else seam.gap_nominal
                source = "measured fit-up (85th percentile)"
            else:
                gap = seam.gap_nominal + 0.5 * seam.gap_tolerance
                source = "drawing + half the fit-up tolerance"

            if gap < 1.0e-3:
                cls, I, v = "tight", 240.0, 5.0e-3
            elif gap < 2.8e-3:
                cls, I, v = "nominal", 225.0, 4.5e-3
            else:
                cls, I, v = "wide", 205.0, 3.6e-3

            segments.append(
                Segment(
                    s_start=s0,
                    s_end=s1,
                    gap_class=cls,
                    I_set=I,
                    v_travel=v,
                    v_wire=melting_rate(I, stickout, c.arc),
                    weave_amp=min(
                        c.control.weave_per_gap * gap, c.robot.weave_amp_max
                    ),
                    rationale=f"gap {gap * 1e3:.1f} mm from {source} -> {cls}",
                )
            )

        cycle = sum((s.s_end - s.s_start) / s.v_travel for s in segments)
        return WeldPlan(
            segments=tuple(segments),
            p_target=c.control.p_target,
            p_lo=c.control.p_lo,
            p_hi=c.control.p_hi,
            estimated_cycle_time=cycle,
            assumptions=(
                "gap per segment is a single number; the real gap varies inside "
                "every segment and the motion layer is expected to handle that",
                "no distortion or heat build-up between passes is modelled",
                "starting parameters only: the adaptive layer will move away "
                "from them, and should",
            ),
        )
