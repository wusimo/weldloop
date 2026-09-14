"""Turning a :class:`WeldPlan` into something the 20 ms layer can read.

A plan is a list of segments in *arc length*.  The motion layer runs in time
and knows where it is only from the robot encoder.  This module is the whole
of the translation, and it is short on purpose: the plan enters the real-time
loop through exactly one function, and every setpoint it carries has been
through :meth:`Nominal.clamped` before the loop starts.

That is the safety argument for letting a language model anywhere near this.
The model writes a table of numbers offline; the table is validated, clamped
and frozen; the loop then reads the table with a table lookup.  Nothing in the
real-time path calls a model, waits on a network, or can be made slow by one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weldloop.config import WeldConfig
from weldloop.control.setpoints import ClampReport, Nominal
from weldloop.planning.task_planner import WeldPlan

__all__ = ["PlanSchedule"]


@dataclass(frozen=True, slots=True)
class PlanSchedule:
    """Pre-clamped per-segment setpoints, looked up by arc length."""

    edges: np.ndarray                    # [m], length n+1, monotone
    nominals: tuple[Nominal, ...]
    labels: tuple[str, ...]
    clamps: tuple[tuple[ClampReport, ...], ...]
    plan: WeldPlan

    @classmethod
    def from_plan(cls, plan: WeldPlan, cfg: WeldConfig) -> "PlanSchedule":
        """Validate and clamp a plan into a lookup table.

        Segments are taken in the order given, their boundaries forced to be
        monotone, and the first/last extended to cover the whole seam so a
        lookup can never fall off either end.
        """
        if not plan.segments:
            raise ValueError("a plan with no segments cannot drive a weld")

        segs = sorted(plan.segments, key=lambda s: s.s_start)
        edges = [float(segs[0].s_start)]
        for s in segs:
            edges.append(max(float(s.s_end), edges[-1] + 1e-6))
        edges[0] = min(edges[0], 0.0)

        noms, labels, clamps = [], [], []
        for s in segs:
            nom, rep = Nominal(
                I_set=s.I_set,
                v_travel=s.v_travel,
                weave_amp=s.weave_amp,
                p_target=s.p_target if s.p_target is not None else plan.p_target,
                p_lo=s.p_lo if s.p_lo is not None else plan.p_lo,
                p_hi=s.p_hi if s.p_hi is not None else plan.p_hi,
                thickness=(
                    s.thickness if s.thickness is not None else cfg.joint.thickness
                ),
            ).clamped(cfg)
            noms.append(nom)
            labels.append(s.gap_class)
            clamps.append(rep)

        return cls(
            edges=np.asarray(edges, dtype=float),
            nominals=tuple(noms),
            labels=tuple(labels),
            clamps=tuple(clamps),
            plan=plan,
        )

    # -- the real-time path ----------------------------------------------
    def index_at(self, s: float) -> int:
        """Which segment covers arc length ``s`` (clamped to the ends)."""
        i = int(np.searchsorted(self.edges, s, side="right")) - 1
        return min(max(i, 0), len(self.nominals) - 1)

    def nominal_at(self, s: float) -> Nominal:
        return self.nominals[self.index_at(s)]

    # -- diagnostics ------------------------------------------------------
    @property
    def n_clamped(self) -> int:
        """How many setpoints the plan asked for that the cell refused."""
        return sum(len(c) for c in self.clamps)

    def clamp_lines(self) -> list[str]:
        out = []
        for i, reps in enumerate(self.clamps):
            for r in reps:
                scale, unit = (1e3, " mm") if r.field != "I_set" else (1.0, " A")
                if r.field == "v_travel":
                    scale, unit = 1e3, " mm/s"
                out.append(f"segment {i}: {r.describe(scale, unit)}")
        return out
