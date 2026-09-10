"""ms-level power-source inner loop.

This layer **models what the inverter already does**, and the demo is careful
not to claim otherwise.  A synergic CV machine already holds its own arc
length and already self-regulates the electrode extension at ~10 ms; adding a
second controller that fights it would be worse than nothing.

What this class does is the small amount that is genuinely left over:

* **Current is commanded through the wire feed.**  Mass conservation pins the
  mean current to the burn-off rate, so asking for 210 A means asking for the
  wire feed that melts at 210 A.  The feed-forward is the inverted burn-off
  law; a slow PI trims the residual error using the measured current.
* **Arc length is commanded through the set voltage.**  The machine's set
  voltage is computed from the requested arc length and current pair, exactly
  as a synergic line does; a PI trims it against the arc length inverted from
  the measured V/I.

Both loops are deliberately slower than the machine's own current regulator.
The point of this layer is to place setpoints, not to chase the arc.

TODO(real-hw): on a real machine this is a handful of fieldbus writes per
cycle.  The gains here would be replaced by the machine's own synergic table.
"""

from __future__ import annotations

from dataclasses import dataclass

from weldloop._fastmath import clip
from weldloop.config import WeldConfig
from weldloop.physics.arc import current_for_melting_rate, melting_rate

__all__ = ["InnerLoopOutput", "InnerLoop"]


@dataclass(frozen=True, slots=True)
class InnerLoopOutput:
    """Setpoints handed to the machine."""

    I_set: float
    v_wire_set: float
    arc_len_set: float


class InnerLoop:
    """PI trim on top of the machine's own regulation."""

    def __init__(self, cfg: WeldConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._i_wire = 0.0
        self._i_arc = 0.0
        self.last = InnerLoopOutput(
            I_set=self.cfg.baseline.I_set,
            v_wire_set=self.cfg.baseline.v_wire,
            arc_len_set=self.cfg.arc.L_arc_ref,
        )

    def update(
        self,
        dt: float,
        *,
        I_cmd: float,
        L_arc_cmd: float,
        I_meas: float,
        L_arc_meas: float,
    ) -> InnerLoopOutput:
        """One inner-loop tick.  ``*_meas`` may be NaN; the loop then coasts."""
        c = self.cfg
        ps, arc = c.power_source, c.arc
        stickout_nom = c.robot.ctwd_nom - L_arc_cmd

        # --- current, commanded through the wire feed ---------------------
        v_wire_ff = melting_rate(I_cmd, stickout_nom, arc)
        if I_meas == I_meas:  # not NaN
            err_I = I_cmd - I_meas
            self._i_wire = clip(self._i_wire + err_I * dt, -400.0, 400.0)
            trim = ps.kp_wire_I * err_I + ps.ki_wire_I * self._i_wire
        else:
            trim = 0.0
        v_wire_set = clip(v_wire_ff + trim, ps.v_wire_min, ps.v_wire_max)

        # --- arc length, commanded through the set voltage ----------------
        if L_arc_meas == L_arc_meas:
            err_L = L_arc_cmd - L_arc_meas
            self._i_arc = clip(self._i_arc + err_L * dt, -0.05, 0.05)
            # gains are quoted per millimetre, which is how a welder thinks
            trim_L = 1e-3 * (ps.kp_arc * err_L * 1e3 + ps.ki_arc * self._i_arc * 1e3)
        else:
            trim_L = 0.0
        arc_len_set = clip(L_arc_cmd + trim_L, ps.L_arc_min, 12.0e-3)

        self.last = InnerLoopOutput(
            I_set=clip(I_cmd, ps.I_min, ps.I_max),
            v_wire_set=v_wire_set,
            arc_len_set=arc_len_set,
        )
        return self.last

    @staticmethod
    def current_for_wire(v_wire: float, stickout: float, cfg: WeldConfig) -> float:
        """Convenience inverse: what current does this wire feed imply?"""
        return current_for_melting_rate(v_wire, stickout, cfg.arc)
