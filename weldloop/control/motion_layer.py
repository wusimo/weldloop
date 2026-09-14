"""The 10-100 ms adaptive torch-motion layer — the focus of this demo.

It consumes the estimator's **mean and covariance** and moves three handles:
travel speed, weave amplitude, and the current/wire-feed setpoint (through the
inner loop).  Everything it does follows from four requirements:

1. **Keep penetration inside the acceptance band.**  Not the mean penetration —
   the whole ``+/- k_sigma * sigma`` interval.  Asking for ``p_hat`` in band
   would let the controller run right up against burn-through whenever the
   estimate happened to be uncertain.
2. **Keep the pool wetting both sidewalls**: ``w >= gap + 2 * margin``, again on
   the lower confidence bound of ``w``.
3. **Fill the joint**: deposited area must cover ``gap * thickness`` plus the
   crown, which is a feed-forward from the previewed gap, trimmed by the
   estimated fill ratio.
4. **Go as fast as the above allows.**  The productivity term speeds the torch
   up in proportion to the *slack* left between the confidence interval and the
   acceptance band.  This is the mechanism that makes uncertainty expensive:
   a wide interval leaves no slack, so an unsure controller simply does not
   speed up.  Conservatism is not a special case bolted on, it falls out.

On top of that sits one hard safety rule: roll the process model forward over
a short horizon with the commanded inputs and the previewed gap, and if the
predicted penetration (plus its uncertainty margin) reaches
``bt_margin * thickness``, immediately speed up, cut current and weave wide.
That path ignores the PI loops entirely.

Note what is *not* here: no vision, no learned policy, no planner.  The layer
is a few hundred lines of interpretable control law, which is what a welding
company can be asked to certify.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from weldloop._fastmath import clip
from weldloop.config import WeldConfig, estimator_config
from weldloop.control.inner_loop import InnerLoop
from weldloop.control.setpoints import Nominal
from weldloop.estimation.ekf import EKFOutput, PoolEKF, SensorSet
from weldloop.estimation.features import FeatureStream, GapTracker
from weldloop.physics.arc import arc_length_from_voltage, melting_rate
from weldloop.physics.melt_pool import MeltPoolModel, PoolInputs
from weldloop.planning.schedule import PlanSchedule
from weldloop.sim.cell import TorchCommand

__all__ = ["MotionDecision", "MotionLayer", "AdaptiveController"]


@dataclass(frozen=True, slots=True)
class MotionDecision:
    """What the motion layer decided, and enough diagnostics to explain it."""

    v_travel: float
    weave_amp: float
    I_cmd: float
    L_arc_cmd: float
    emergency: bool
    reason: str
    p_hat: float
    p_ucb: float
    p_lcb: float
    p_pred: float
    gap_ff: float
    slack: float
    v_wire_needed: float
    segment: int = -1


class MotionLayer:
    """The control law, with no I/O — so it can be tested on its own."""

    def __init__(self, cfg: WeldConfig) -> None:
        self.cfg = cfg
        self.model = MeltPoolModel(estimator_config(cfg))
        self.reset()

    def reset(self) -> None:
        c = self.cfg
        self._nom_default = Nominal.from_config(c)
        self._integ = 0.0
        self.v_cmd = c.baseline.v_travel
        self.weave_cmd = 0.0
        self.I_cmd = c.baseline.I_set
        self.last: MotionDecision | None = None

    # -- burn-through look-ahead -----------------------------------------
    def predict_penetration(
        self, est: EKFOutput, u: PoolInputs, horizon: float, n_sub: int = 10
    ) -> float:
        """Roll the process model forward under the commanded inputs."""
        x = est.x.copy()
        arr = u.to_array()
        h = horizon / n_sub
        for _ in range(n_sub):
            x = self.model.step_array(x, arr, h)
        return float(x[2])

    # -- the law ----------------------------------------------------------
    def update(
        self,
        dt: float,
        est: EKFOutput,
        *,
        gap_ff: float,
        v_wire_now: float,
        thickness: float,
        nom: Nominal | None = None,
    ) -> MotionDecision:
        """One motion-layer decision.

        Handle allocation, which is the substance of this layer:

        ``current``
            Primary penetration authority.  A PI tracks ``p_target``, with the
            band-violation terms weighted ``w_safety`` times heavier than
            target tracking, so safety wins any argument with productivity.
        ``travel speed``
            The productivity handle.  It is pushed up only while the whole
            confidence interval sits inside the acceptance band **and** the
            current loop still has headroom to pay for the extra speed; it is
            hard-limited from above by the rate at which filler can fill the
            previewed gap.
        ``weave``
            Bridges the gap, reaches both sidewalls, and relieves
            over-penetration.

        ``nom`` is the operating point to regulate around — nominal current,
        travel speed, weave floor, and the acceptance band.  It defaults to the
        procedure's single global setting; a plan supplies a different one per
        segment.  Whatever it is, it has already been through
        :meth:`Nominal.clamped`, so this function never sees a setpoint the
        machine cannot produce.
        """
        c = self.cfg
        ctl, rb = c.control, c.robot
        if nom is None:
            nom = self._nom_default

        p_hat, sig_p = est.penetration, est.penetration_std
        w_hat, sig_w = est.pool_width, est.pool_width_std
        margin_p = ctl.k_sigma * sig_p
        p_ucb, p_lcb = p_hat + margin_p, p_hat - margin_p

        # --- band accounting -----------------------------------------------
        e_hi = max(p_ucb - nom.p_hi, 0.0)          # confidence interval too deep
        e_lo = max(nom.p_lo - p_lcb, 0.0)          # confidence interval too shallow
        slack = min(nom.p_hi - p_ucb, p_lcb - nom.p_lo)   # may be negative
        e_track = (nom.p_target - p_hat) / nom.p_target
        e_band = ctl.w_safety * (e_lo - e_hi) / nom.p_target
        e_norm = e_track + e_band

        # --- current: primary penetration authority -------------------------
        I_nom = nom.I_set
        I_unsat = I_nom * (1.0 + ctl.kp_I * e_norm + ctl.ki_I * self._integ)
        I_cmd = clip(I_unsat, ctl.I_min_cmd, ctl.I_max_cmd)
        # rate-limit the command: the estimate is noisy tick to tick and an
        # inverter that hunts over tens of amps at 50 Hz is worse than useless
        dI = clip(I_cmd - self.I_cmd, -ctl.I_slew * dt, ctl.I_slew * dt)
        I_cmd = clip(self.I_cmd + dI, ctl.I_min_cmd, ctl.I_max_cmd)
        # integrate only when the actuator can still act on it (anti-windup)
        saturated = (I_unsat > ctl.I_max_cmd and e_norm > 0.0) or (
            I_unsat < ctl.I_min_cmd and e_norm < 0.0
        )
        if not saturated:
            self._integ = clip(self._integ + e_norm * dt, -0.8, 0.8)
        else:
            self._integ *= math.exp(-dt / ctl.tau_integ)

        # --- weave: bridge the gap, reach both sidewalls, relieve depth ------
        a_gap = max(ctl.weave_per_gap * max(gap_ff, 0.0), nom.weave_amp)
        need_width = max(gap_ff, 0.0) + 2.0 * c.joint.sidewall_margin
        w_lcb = w_hat - ctl.k_sigma * sig_w
        a_width = 0.5 * max(need_width - w_lcb, 0.0)
        a_pen = ctl.k_weave_pen * (e_hi / nom.p_target) * rb.weave_amp_max
        weave_cmd = clip(max(a_gap, a_width) + a_pen, 0.0, rb.weave_amp_max)

        # --- speed: productivity, gated by band slack and current headroom ---
        slack_norm = slack / nom.p_target
        headroom = (ctl.I_max_cmd - I_cmd) / (ctl.I_max_cmd - ctl.I_min_cmd)
        push = ctl.k_prod * min(slack_norm, headroom - ctl.I_headroom)
        v_want = nom.v_travel * (1.0 + push - ctl.kp_v * (e_lo - e_hi) / nom.p_target)

        # ...and hard-limited by what the filler can actually fill
        A_req = max(gap_ff, 0.0) * thickness + c.joint.A_reinf
        v_wire_expected = melting_rate(I_cmd, rb.ctwd_nom - c.arc.L_arc_ref, c.arc)
        v_fill_max = (
            c.consumable.eta_dep * c.consumable.area * v_wire_expected
            / max(A_req * ctl.fill_target, 1e-12)
        )
        v_fill_max *= 1.0 - ctl.kp_wire * clip(1.0 - est.fill, -0.3, 0.3)
        v_target = clip(min(v_want, v_fill_max), rb.v_travel_min, rb.v_travel_max)

        dv = clip(v_target - self.v_cmd, -ctl.v_slew * dt, ctl.v_slew * dt)
        v_cmd = clip(self.v_cmd + dv, rb.v_travel_min, rb.v_travel_max)

        # --- hard safety: predicted burn-through -----------------------------
        emergency = False
        reason = "nominal"
        u_pred = PoolInputs(
            I=I_cmd,
            V=27.0,
            v_travel=v_cmd,
            v_wire=v_wire_expected,
            gap=max(gap_ff, 0.0),
            thickness=thickness,
            weave_amp=weave_cmd,
        )
        # A safety monitor must never predict something safer than what it
        # already believes: the model is known to be biased, and the filter is
        # holding the state up against it on measurement evidence.
        p_pred = max(self.predict_penetration(est, u_pred, ctl.horizon), p_hat)
        trip = ctl.bt_margin * thickness
        if p_pred + margin_p >= trip or p_ucb >= trip:
            emergency = True
            reason = "predicted burn-through"
            v_cmd = clip(v_cmd * ctl.bt_speed_boost, rb.v_travel_min, rb.v_travel_max)
            I_cmd = clip(I_cmd * ctl.bt_current_cut, ctl.I_min_cmd, ctl.I_max_cmd)
            weave_cmd = rb.weave_amp_max
            self._integ = min(self._integ, 0.0)  # do not fight the trip

        self.v_cmd, self.weave_cmd, self.I_cmd = v_cmd, weave_cmd, I_cmd
        self.last = MotionDecision(
            v_travel=v_cmd,
            weave_amp=weave_cmd,
            I_cmd=I_cmd,
            L_arc_cmd=c.arc.L_arc_ref,
            emergency=emergency,
            reason=reason,
            p_hat=p_hat,
            p_ucb=p_ucb,
            p_lcb=p_lcb,
            p_pred=p_pred,
            gap_ff=gap_ff,
            slack=slack,
            v_wire_needed=v_wire_expected,
        )
        return self.last


# --------------------------------------------------------------------------
class AdaptiveController:
    """Estimator + motion layer + inner loop, wired to the cell's clocks.

    Three timescales, as the proposal describes them:

    * every simulation tick, the 5 kHz V/I samples are pushed into the feature
      ring buffer (a DAQ callback on real hardware);
    * every ``power_source.inner_dt`` (2 ms), the inner loop refreshes the
      electrical setpoints;
    * every ``control.dt`` (20 ms), features are extracted, the EKF is updated,
      and the motion layer re-decides.

    ``sensor_set`` selects which measurements the estimator may use, which is
    how the RGB-only control arm is built: same law, worse state.

    ``plan`` is optional feed-forward from the seconds-to-minutes layer: a
    :class:`~weldloop.planning.task_planner.WeldPlan` whose segments become the
    nominal operating point the law regulates around, looked up by encoder
    position.  Without it the whole seam is regulated around the one operating
    point in the config, which is what a procedure sheet gives you.  The plan
    changes the *starting point per segment*; it does not get a vote once the
    arc is lit.
    """

    def __init__(
        self,
        cfg: WeldConfig,
        *,
        sensor_set: str = "all",
        residual=None,
        plan=None,
        name: str = "adaptive",
    ) -> None:
        self.cfg = cfg
        self.name = name
        self.sensor_set = SensorSet.named(sensor_set)
        self.residual = residual
        self.plan = plan
        self.reset(cfg)

    # -- lifecycle -------------------------------------------------------
    def reset(self, cfg: WeldConfig | None = None) -> None:
        cfg = cfg or self.cfg
        self.cfg = cfg
        self.features = FeatureStream(cfg)
        self.gaps = GapTracker(default_gap=0.0)
        self.ekf = PoolEKF(cfg, self.sensor_set, residual=self.residual)
        self.motion = MotionLayer(cfg)
        self.inner = InnerLoop(cfg)
        self.schedule = (
            PlanSchedule.from_plan(self.plan, cfg) if self.plan is not None else None
        )
        self._V_ema = math.nan
        self._I_ema = math.nan
        self._last_prof_s = -1.0
        self._t_last_ctl = 0.0
        self.history: list[dict] = []
        self.est: EKFOutput | None = None
        self._cmd = TorchCommand.from_config(cfg)

    # -- 5 kHz: raw stream ------------------------------------------------
    def on_samples(self, t: float, samples) -> None:
        """DAQ callback: push V/I into the feature buffer, profiler into the map."""
        for smp in samples:
            ch = smp.channels
            if "ps_V" in ch:
                self.features.push(ch["ps_V"], ch["ps_I"], ch.get("ps_short", 0.0))
                tau = 0.02
                a = min(1.0 / self.cfg.sensors.f_master / tau, 1.0)
                self._V_ema = (
                    ch["ps_V"] if self._V_ema != self._V_ema
                    else self._V_ema + a * (ch["ps_V"] - self._V_ema)
                )
                self._I_ema = (
                    ch["ps_I"] if self._I_ema != self._I_ema
                    else self._I_ema + a * (ch["ps_I"] - self._I_ema)
                )
            elif smp.valid and "prof_gap" in ch:
                g, s_look = ch["prof_gap"], ch["prof_lead_s"]
                if g == g and s_look == s_look:
                    self.gaps.push(s_look, g)

    # -- 2 ms: inner loop -------------------------------------------------
    def fast_update(self, t: float, obs) -> TorchCommand | None:
        """Refresh only the electrical setpoints; motion commands are held."""
        c = self.cfg
        d = self.motion.last
        I_cmd = d.I_cmd if d is not None else (
            self.schedule.nominal_at(0.0).I_set if self.schedule is not None
            else c.baseline.I_set
        )
        L_cmd = d.L_arc_cmd if d is not None else c.arc.L_arc_ref
        L_meas = math.nan
        if self._V_ema == self._V_ema:
            L_meas = arc_length_from_voltage(
                self._V_ema, self._I_ema, c.arc.stickout, c.arc, c.consumable
            )
        out = self.inner.update(
            c.power_source.inner_dt,
            I_cmd=I_cmd,
            L_arc_cmd=L_cmd,
            I_meas=self._I_ema,
            L_arc_meas=L_meas,
        )
        self._cmd = TorchCommand(
            I_set=out.I_set,
            v_wire_set=out.v_wire_set,
            arc_len_set=out.arc_len_set,
            v_travel=self._cmd.v_travel,
            weave_amp=self._cmd.weave_amp,
            torch_angle=c.robot.torch_angle,
            ctwd=c.robot.ctwd_nom,
        )
        return self._cmd

    # -- 20 ms: estimate and decide ---------------------------------------
    def update(self, t: float, obs) -> TorchCommand | None:
        c = self.cfg
        if not self.features.due(t):
            return self._cmd  # buffer not full yet: run the baseline command

        feats = self.features.extract(t)
        s_now = obs.get("rb_s", 0.0)
        v_now = obs.get("rb_v_travel", c.baseline.v_travel)
        v_wire_now = obs.get("ps_v_wire", c.baseline.v_wire)

        # the plan's segment has to be resolved *before* the estimator input is
        # built: the process model needs the same plate thickness the safety
        # monitor is using, or the filter and the monitor are describing
        # different joints.
        seg = -1
        nom = None
        if self.schedule is not None:
            seg = self.schedule.index_at(s_now)
            nom = self.schedule.nominals[seg]
        thickness = nom.thickness if nom is not None else c.joint.thickness

        gap_now = self.gaps.gap_at(s_now)
        # act on the gap that will be under the arc a short time from now
        gap_ff = self.gaps.gap_at(s_now + v_now * c.control.gap_lead_time)
        if not self.sensor_set.use_profiler:
            gap_now = gap_ff = 0.0  # no scanner: the procedure's assumed fit-up

        u = np.array(
            [
                feats.I_mean if feats.valid else c.baseline.I_set,
                feats.V_mean if feats.valid else 27.0,
                v_now,
                v_wire_now,
                gap_now,
                thickness,
                self._cmd.weave_amp,
            ]
        )
        meas = {
            "f_ripple": feats.f_ripple,
            "a_ripple": feats.a_ripple,
            "f_sc": feats.f_sc,
            "L_arc_est": feats.L_arc_est,
            "ir_T_peak": obs.get("ir_T_peak", math.nan)
            if obs.age("ir_T_peak") < c.control.dt else math.nan,
            "ir_pool_width": obs.get("ir_pool_width", math.nan)
            if obs.age("ir_pool_width") < c.control.dt else math.nan,
            "rgb_pool_width": obs.get("rgb_pool_width", math.nan)
            if obs.age("rgb_pool_width") < c.control.dt else math.nan,
        }
        dt = max(t - self._t_last_ctl, 1e-6)
        self._t_last_ctl = t
        self.est = self.ekf.step(t, u, meas, dt=min(dt, 5.0 * c.ekf.dt))

        decision = self.motion.update(
            dt, self.est, gap_ff=gap_ff, v_wire_now=v_wire_now,
            thickness=thickness, nom=nom,
        )
        decision = replace(decision, segment=seg)
        self.motion.last = decision
        self._cmd = TorchCommand(
            I_set=self._cmd.I_set,
            v_wire_set=self._cmd.v_wire_set,
            arc_len_set=self._cmd.arc_len_set,
            v_travel=decision.v_travel,
            weave_amp=decision.weave_amp,
            torch_angle=c.robot.torch_angle,
            ctwd=c.robot.ctwd_nom,
        )
        self.history.append(
            {
                "t": t, "s": s_now, "p_hat": decision.p_hat,
                "p_std": self.est.penetration_std, "p_ucb": decision.p_ucb,
                "p_lcb": decision.p_lcb, "p_pred": decision.p_pred,
                "gap_ff": decision.gap_ff, "v_travel": decision.v_travel,
                "weave": decision.weave_amp, "I_cmd": decision.I_cmd,
                "emergency": float(decision.emergency), "slack": decision.slack,
                "segment": float(seg),
                "w_hat": self.est.pool_width, "fill_hat": self.est.fill,
            }
        )
        return self._cmd
