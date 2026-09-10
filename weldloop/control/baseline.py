"""Fixed-parameter controller — the thing the proposal is arguing against.

This is what a welding cell does today: an operator picks a current, a voltage
and a travel speed from a procedure sheet, the robot runs the seam, and
nothing reacts to what the joint actually looks like.  It is a fair
comparison, not a straw man: the parameters are the ones that produce a good
weld at the *nominal* fit-up, which is exactly how a real procedure is
qualified.

Its failure mode on a variable-gap seam is the whole point.  Where the gap
opens, the same heat input digs deeper into less metal and the pool drops out;
where the gap closes, the same wire feed over-fills.
"""

from __future__ import annotations

from weldloop.config import WeldConfig
from weldloop.sim.cell import TorchCommand

__all__ = ["BaselineController"]


class BaselineController:
    """Issues one command and never changes it."""

    name = "baseline"

    def __init__(self, cfg: WeldConfig) -> None:
        self.cfg = cfg
        self.reset(cfg)

    def reset(self, cfg: WeldConfig | None = None) -> None:
        self.cfg = cfg or self.cfg
        b = self.cfg.baseline
        self._cmd = TorchCommand(
            I_set=b.I_set,
            v_wire_set=b.v_wire,
            arc_len_set=self.cfg.arc.L_arc_ref,
            v_travel=b.v_travel,
            weave_amp=b.weave_amp,
            torch_angle=self.cfg.robot.torch_angle,
            ctwd=self.cfg.robot.ctwd_nom,
        )
        self.history: list[dict] = []

    def update(self, t: float, obs) -> TorchCommand | None:
        return self._cmd
