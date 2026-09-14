"""Open-loop execution of a plan: parameters change per segment, nothing else.

This is the honest middle ground between :class:`BaselineController` (one
parameter set for the whole seam) and :class:`AdaptiveController` (a closed
loop at 50 Hz).  It is what you get from a *good offline plan and no process
feedback at all* — which is what most "smart welding" demos actually are, and
which is the arm the adaptive layer has to beat to be worth the hardware.

It reads exactly one thing from the cell: the robot encoder, ``rb_s``.  Every
cell already has that.  No arc signal, no camera, no estimator.
"""

from __future__ import annotations

from weldloop.config import WeldConfig
from weldloop.planning.schedule import PlanSchedule
from weldloop.physics.arc import melting_rate
from weldloop.sim.cell import TorchCommand

__all__ = ["PlannedController"]


class PlannedController:
    """Switches to each segment's parameters as the encoder crosses into it."""

    def __init__(self, cfg: WeldConfig, plan, *, name: str = "planned") -> None:
        self.cfg = cfg
        self.name = name
        self.plan = plan
        self.reset(cfg)

    def reset(self, cfg: WeldConfig | None = None) -> None:
        self.cfg = cfg or self.cfg
        self.schedule = PlanSchedule.from_plan(self.plan, self.cfg)
        self._seg = -1
        self.history: list[dict] = []
        self._cmd = self._command_for(0)

    def _command_for(self, i: int) -> TorchCommand:
        c = self.cfg
        nom = self.schedule.nominals[i]
        stickout = c.robot.ctwd_nom - c.arc.L_arc_ref
        return TorchCommand(
            I_set=nom.I_set,
            v_wire_set=melting_rate(nom.I_set, stickout, c.arc),
            arc_len_set=c.arc.L_arc_ref,
            v_travel=nom.v_travel,
            weave_amp=nom.weave_amp,
            torch_angle=c.robot.torch_angle,
            ctwd=c.robot.ctwd_nom,
        )

    def update(self, t: float, obs) -> TorchCommand | None:
        i = self.schedule.index_at(obs.get("rb_s", 0.0))
        if i != self._seg:
            self._seg = i
            self._cmd = self._command_for(i)
            self.history.append(
                {"t": t, "s": obs.get("rb_s", 0.0), "segment": float(i),
                 "I_cmd": self._cmd.I_set, "v_travel": self._cmd.v_travel,
                 "weave": self._cmd.weave_amp}
            )
        return self._cmd
