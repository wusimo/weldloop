"""Extended Kalman filter over the melt-pool state.

Process model
-------------
``physics/melt_pool.py``, built from :func:`config.estimator_config` — i.e. a
*deliberately mismatched* copy of the plant's coefficients.  This matters more
than it looks: if the estimator ran the simulator's own model the filter would
be solving a problem nobody has, and the learned residual would have nothing to
correct.  The mismatch stands in for the parameter error left after
identifying a reduced-order model from a finite amount of real weld data.

Inputs (known, not estimated)
-----------------------------
Current and voltage from the V/I features, travel speed from the robot encoder,
wire feed from the drive tacho, weave amplitude from the command, plate
thickness from the job.  The **root gap** comes from the look-ahead profiler
via :class:`~weldloop.estimation.features.GapTracker`, delayed to the arc.

That last point is worth stating to the customer: the profiler is not a
*measurement* of the state in the filter sense — it is an **input** to the
process model.  A wrong gap does not get corrected by innovation, it corrupts
the prediction.  This is why the "V/I only" ablation is so much worse than
"V/I + profiler" even though the profiler never observes the pool.

Measurements
------------
=================  ==========================================  =============
channel            model                                       reports
=================  ==========================================  =============
``f_ripple``       ``pool_oscillation(w, ...)[0]``             pool width
``a_ripple``       ``k_ripple * pool_oscillation(...)[1]``     penetration
``f_sc``           ``short_circuit_rate(I, L_est, a_osc)``     both, weakly
``ir_T_peak``      ``T_pool``                                  temperature
``ir_pool_width``  ``w``                                       pool width
``rgb_pool_width`` ``w``                                       pool width (badly)
=================  ==========================================  =============

Updates are applied **sequentially as scalars**.  That costs nothing here, it
is numerically better behaved than one stacked update, and it makes a dropout
a non-event: a NaN measurement is simply skipped, and the covariance grows
until something else arrives.

Jacobians are numerical (forward differences).  The process model is 4 states
of cheap algebra, the filter runs at 50 Hz, and an analytic Jacobian of a model
whose coefficients will be re-identified from real data is false economy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from weldloop.config import WeldConfig, estimator_config
from weldloop.physics.arc import (
    mean_arc_length,
    pool_oscillation,
    short_circuit_rate,
)
from weldloop.physics.melt_pool import MeltPoolModel, PoolInputs, PoolState

__all__ = ["SensorSet", "EKFOutput", "PoolEKF"]

#: Named ablations for the Phase 3 table.
SENSOR_SETS: dict[str, tuple[str, ...]] = {
    "vi": ("f_ripple", "a_ripple", "f_sc"),
    "vi+profiler": ("f_ripple", "a_ripple", "f_sc"),
    "all": ("f_ripple", "a_ripple", "f_sc", "ir_T_peak", "ir_pool_width"),
    "rgb": ("rgb_pool_width",),
}
#: which ablations are allowed to use the profiler as a process-model input
USES_PROFILER: dict[str, bool] = {
    "vi": False, "vi+profiler": True, "all": True, "rgb": False,
}


@dataclass(frozen=True, slots=True)
class SensorSet:
    """One ablation: which measurements are used, and whether the gap is known."""

    name: str
    channels: tuple[str, ...]
    use_profiler: bool

    @staticmethod
    def named(name: str) -> "SensorSet":
        return SensorSet(name, SENSOR_SETS[name], USES_PROFILER[name])


@dataclass(slots=True)
class EKFOutput:
    """Filter state after one update."""

    t: float
    x: np.ndarray
    P: np.ndarray
    n_used: int = 0
    innovations: dict[str, float] = field(default_factory=dict)

    @property
    def penetration(self) -> float:
        return float(self.x[2])

    @property
    def penetration_std(self) -> float:
        return float(math.sqrt(max(self.P[2, 2], 0.0)))

    @property
    def pool_width(self) -> float:
        return float(self.x[1])

    @property
    def pool_width_std(self) -> float:
        return float(math.sqrt(max(self.P[1, 1], 0.0)))

    @property
    def T_pool(self) -> float:
        return float(self.x[0])

    @property
    def fill(self) -> float:
        return float(self.x[3])

    @property
    def fill_std(self) -> float:
        return float(math.sqrt(max(self.P[3, 3], 0.0)))

    def as_state(self) -> PoolState:
        return PoolState.from_array(self.x)


class PoolEKF:
    """EKF over ``[T_pool, w, p, f]``.

    Usage is the same offline and online::

        ekf = PoolEKF(cfg, SensorSet.named("all"))
        out = ekf.step(t, u, measurements)
    """

    #: state bounds; the filter is clamped to physically possible values
    _LO = np.array([1700.0, 1.0e-3, 0.2e-3, 0.0])
    _HI = np.array([3200.0, 40.0e-3, 20.0e-3, 4.0])

    def __init__(
        self,
        cfg: WeldConfig,
        sensor_set: SensorSet | str = "all",
        *,
        residual=None,
        use_true_model: bool = False,
    ) -> None:
        self.cfg = cfg
        self.est_cfg = cfg if use_true_model else estimator_config(cfg)
        self.model = MeltPoolModel(self.est_cfg)
        self.sensors = (
            sensor_set if isinstance(sensor_set, SensorSet) else SensorSet.named(sensor_set)
        )
        #: optional learned correction, see ``estimation/residual.py``
        self.residual = residual
        # The V/I features come from a sliding window that is much longer than
        # the filter step, so consecutive feature vectors share most of their
        # samples and their errors are strongly correlated.  Treating them as
        # independent would let the covariance shrink by sqrt(N) on data that
        # contains only N*dt/window independent looks -- which is exactly how a
        # filter ends up confidently wrong.  Inflate R by the overlap factor.
        self.overlap = max(cfg.estimator_window / cfg.ekf.dt, 1.0)
        # The root gap is an INPUT to the process model, not a measurement, so an
        # error in it is never corrected by innovation -- it corrupts the
        # prediction.  Its uncertainty therefore has to be propagated into P
        # explicitly, otherwise a cell with no profiler produces a filter that is
        # confidently wrong, which is worse than one that admits it.
        e = cfg.ekf
        self.gap_std = e.gap_std_profiler if self.sensors.use_profiler else e.gap_std_blind
        self.reset()

    # -- lifecycle -------------------------------------------------------
    def reset(self, x0: np.ndarray | None = None) -> None:
        c = self.cfg
        if x0 is None:
            u = PoolInputs(
                I=c.baseline.I_set,
                V=27.0,
                v_travel=c.baseline.v_travel,
                v_wire=c.baseline.v_wire,
                gap=0.0,
                thickness=c.joint.thickness,
            )
            x0 = self.model.steady_state(u).to_array()
        self.x = np.asarray(x0, dtype=float).copy()
        e = c.ekf
        self.P = np.diag(np.array([e.p0_T, e.p0_w, e.p0_p, e.p0_f]) ** 2)
        self._Q1 = np.array([e.q_T, e.q_w, e.q_p, e.q_f]) ** 2  # per second
        self.n_steps = 0

    # -- prediction ------------------------------------------------------
    def _propagate(self, x: np.ndarray, u: np.ndarray, dt: float, n_sub: int = 4) -> np.ndarray:
        """Integrate the process model over ``dt`` with sub-steps.

        The filter runs at 50 Hz while the model's fastest time constant is
        20 ms, so a single Euler step of 20 ms would be marginal; four sub-steps
        make the discretisation error irrelevant without meaningful cost.
        """
        h = dt / n_sub
        for _ in range(n_sub):
            x = self.model.step_array(x, u, h)
        if self.residual is not None:
            x = x + self.residual.correction(x, u, dt)
        return x

    def predict(self, u: np.ndarray, dt: float) -> None:
        x0 = self.x
        f0 = self._propagate(x0, u, dt)

        # forward-difference Jacobian, scaled per state
        F = np.empty((4, 4))
        eps = np.array([1.0, 5.0e-6, 5.0e-6, 5.0e-3])
        for j in range(4):
            xp = x0.copy()
            xp[j] += eps[j]
            F[:, j] = (self._propagate(xp, u, dt) - f0) / eps[j]

        # sensitivity of the prediction to the unknown root gap
        eps_gap = 5.0e-6
        u_gap = u.copy()
        u_gap[4] += eps_gap
        Jg = (self._propagate(x0, u_gap, dt) - f0) / eps_gap
        amplify = max(self.cfg.ekf.gap_persistence / dt, 1.0)
        Q_gap = np.outer(Jg, Jg) * (self.gap_std**2) * amplify

        self.x = f0
        self.P = F @ self.P @ F.T + np.diag(self._Q1 * dt) + Q_gap

    # -- measurement models ----------------------------------------------
    def _h(self, x: np.ndarray, channel: str, aux: dict[str, float]) -> float:
        c = self.est_cfg
        T, w, p, _ = x
        superheat = max(T - c.material.T_m, 0.0) / (c.material.T_m - c.material.T_0)
        if channel == "f_ripple":
            return pool_oscillation(w, p, superheat, c.arc, c.material)[0]
        if channel == "a_ripple":
            return self.cfg.ekf.k_ripple * pool_oscillation(
                w, p, superheat, c.arc, c.material
            )[1]
        if channel == "f_sc":
            a_osc = pool_oscillation(w, p, superheat, c.arc, c.material)[1]
            L = aux.get("L_arc_est", c.arc.L_arc_ref)
            if not np.isfinite(L) or L <= 0.0:
                L = mean_arc_length(c.arc.L_arc_ref, p, c.arc)
            return short_circuit_rate(aux.get("I", c.baseline.I_set), L, a_osc, c.arc)
        if channel == "ir_T_peak":
            return float(T)
        if channel in ("ir_pool_width", "rgb_pool_width"):
            return float(w)
        raise KeyError(channel)

    #: channels produced by a sliding window, hence correlated step to step
    _WINDOWED = frozenset({"f_ripple", "a_ripple", "f_sc"})

    def _R(self, channel: str) -> float:
        e = self.cfg.ekf
        base = {
            "f_ripple": e.r_ripple_f**2,
            "a_ripple": e.r_ripple_a**2,
            "f_sc": e.r_fsc**2,
            "ir_T_peak": e.r_ir_T**2,
            "ir_pool_width": e.r_ir_w**2,
            "rgb_pool_width": e.r_rgb_w**2,
        }[channel]
        return base * self.overlap if channel in self._WINDOWED else base

    def _scalar_update(self, z: float, channel: str, aux: dict[str, float]) -> float:
        """One sequential scalar update.  Returns the innovation."""
        h0 = self._h(self.x, channel, aux)
        if not np.isfinite(h0):
            return math.nan
        H = np.empty(4)
        eps = np.array([1.0, 5.0e-6, 5.0e-6, 5.0e-3])
        for j in range(4):
            xp = self.x.copy()
            xp[j] += eps[j]
            hp = self._h(xp, channel, aux)
            H[j] = (hp - h0) / eps[j] if np.isfinite(hp) else 0.0

        R = self._R(channel)
        PHt = self.P @ H
        S = float(H @ PHt + R)
        if S <= 0.0 or not np.isfinite(S):
            return math.nan
        K = PHt / S
        innov = float(z - h0)
        self.x = self.x + K * innov
        # Joseph form keeps P symmetric positive-definite through many updates
        I_KH = np.eye(4) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R
        return innov

    # -- one filter step -------------------------------------------------
    def step(
        self, t: float, u: np.ndarray, measurements: dict[str, float], dt: float | None = None
    ) -> EKFOutput:
        """Predict with ``u``, then apply whichever measurements are present.

        ``measurements`` may contain NaN for channels that did not sample; those
        are skipped, which is exactly what should happen on a dropout.
        """
        dt = self.cfg.ekf.dt if dt is None else dt
        self.predict(u, dt)

        aux = {
            "I": float(u[0]),
            "L_arc_est": float(measurements.get("L_arc_est", math.nan)),
        }
        innovations: dict[str, float] = {}
        n_used = 0
        for channel in self.sensors.channels:
            z = measurements.get(channel, math.nan)
            if z is None or not np.isfinite(z):
                continue
            innov = self._scalar_update(float(z), channel, aux)
            if np.isfinite(innov):
                innovations[channel] = innov
                n_used += 1

        self.x = np.clip(self.x, self._LO, self._HI)
        self.P = 0.5 * (self.P + self.P.T)
        self.n_steps += 1
        return EKFOutput(
            t=t, x=self.x.copy(), P=self.P.copy(), n_used=n_used, innovations=innovations
        )
