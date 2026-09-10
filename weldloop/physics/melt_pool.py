"""Reduced-order melt-pool model.

WHAT THIS IS
------------
A **control-oriented reduced-order model (ROM)** of the weld pool: four
lumped states integrated at the simulation rate.  It is *not* a CFD or
thermal-FE model and must not be read as one.  Its structure is borrowed from
the classical analytic solutions (Rosenthal 1946; Goldak 1984 — see
``heat_source.py``), and every coefficient in it is a lumped calibration
constant to be re-identified from real weld data.

Why a ROM: the estimator (``estimation/ekf.py``) has to run this model, plus
its Jacobian, at 50 Hz on a laptop CPU; and the controller has to reason about
its states.  A model that cannot be inverted in real time is useless as a
process model, however accurate it is offline.

STATE
-----
    x = [T_pool, w, p, f]

    T_pool [K]  mean melt-pool temperature (>= T_m while the pool exists)
    w      [m]  pool width at the top surface
    p      [m]  penetration depth, from the top surface downwards
    f      [-]  gap fill ratio = deposited area / area required to fill joint

INPUTS
------
    u = [I, V, v_travel, v_wire, gap, thickness, weave_amp]

DERIVED GEOMETRY (algebraic)
----------------------------
    A      = pi/4 * w * p        fused cross-section (half-ellipse)     [m^2]
    l_eff  = kappa_L * w         pool length                            [m]
    Vol    = A * l_eff           molten volume                          [m^3]
    AR     = w / p               bead aspect ratio                      [-]
    A_surf = pi/4 * w * l_eff    free top surface                       [m^2]

EQUATIONS
---------
1. Power book-keeping::

       q_arc  = eta_arc * V * I
       q_cond = C_cond * k * w * (T - T_0) * kappa_gap
       q_conv = h_conv * A_surf * (T - T_0)
       q_rad  = eps * sigma_SB * A_surf * (T^4 - T_0^4)
       q_net  = q_arc - q_cond - q_conv - q_rad

   with ``kappa_gap = 1 - k_gap_cond * min(gap/w, 1)``: a root gap removes the
   heat sink directly under the arc, so less power is conducted away and the
   pool runs hotter and deeper.  This is the primary gap -> burn-through
   mechanism in the model.

   The net power is split by a single lumped melting efficiency::

       q_fus = eta_melt * max(q_net, 0)     creates melt
       q_sh  = (1 - eta_melt) * q_net       superheats the melt

2. Fusion / volume balance.  ``rho * h_m * d(Vol)/dt = q_fus - rho*h_m*A*v``
   with ``Vol = A * l_eff`` rearranges exactly to a first-order relaxation::

       dA/dt = (A_ss - A) / tau_A,   A_ss  = q_fus / (rho * h_m * v)
                                     tau_A = l_eff / v   (pool residence time)

   The equilibrium ``A_ss`` is the textbook melting-efficiency relation.  The
   residence-time constant is **capped** at ``tau_A_max``: the single volume
   state cannot represent the fact that the near-arc root responds much faster
   than the pool as a whole.  Capping changes the transient only, never the
   equilibrium.  CONFIGURABLE.

3. Aspect ratio.  Higher current digs (arc pressure), a root gap lets the arc
   root descend into the joint::

       AR_tgt  = AR_0 * (I_ref/I)^n_I * (v_ref/v)^n_v * mode_gain
                      * (1 + k_weave * 2*a_weave / w)
                      / (1 + k_gap_ar * min(gap/w, 1))

   Weaving is represented only through the aspect ratio: it spreads the same
   heat input over a wider track, so the bead gets wider and shallower while
   the energy per unit weld length is unchanged.  A ROM with one lumped pool
   cannot resolve the within-cycle oscillation, and does not try to.
       dAR/dt  = (AR_tgt - AR) / tau_ar

4. Recovering w and p.  With ``A = pi/4*w*p`` and ``AR = w/p``::

       S = (4/pi) * dA/dt ;  D = p^2 * dAR/dt
       dw/dt = (S + D) / (2p) ;   dp/dt = (S - D) / (2w)

5. Superheat.  ``rho*c_p*Vol*dT/dt`` = q_sh, less the superheat carried out by
   metal solidifying behind the torch, less the power that melts and heats the
   incoming wire::

       C_th * dT/dt = q_sh - rho*c_p*(T - T_m)*A*v - m_wire*(c_p*(T-T_0) + L_f)

   ``C_th`` is capped the same way ``tau_A`` is (``tau_T_max``), equilibrium
   unchanged.

6. Gap fill (mass conservation on the wire)::

       A_dep = eta_dep * A_wire * v_wire / v
       A_req = gap * thickness + A_reinf
       f_ss  = clip(A_dep / A_req, 0, f_max);  df/dt = (f_ss - f) / tau_f

   ``A`` above is the *base-metal* fusion area; deposited filler is accounted
   for separately through ``f``.

DEFECTS (flags, not states)
---------------------------
Burn-through if either

  (a) ``p >= beta_bt * thickness`` — melted straight through, or
  (b) the joint is not yet filled (``f < 1``) and the unsupported molten span
      at the root exceeds the surface-tension bridging limit
      ``w_crit = c_st * sqrt(gamma / (rho g))``.  Criterion (b) is what
      actually fires on a wide gap, and it is the physically right story:
      the pool drops out before the arc has melted the full thickness.

Lack of fusion if ``p < p_min`` (cold), or ``w < gap + 2*sidewall_margin``
(the pool never wets both sidewalls), or ``f < f_min`` (underfilled).

DETERMINISM
-----------
Nothing in this module is stochastic.  Given the same state, inputs and dt,
``MeltPoolModel.step`` returns bit-identical results.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from weldloop._fastmath import clip
from weldloop.config import SIGMA_SB, TransferMode, WeldConfig
from weldloop.physics.arc import classify_transfer_mode

__all__ = [
    "PoolState",
    "PoolInputs",
    "PoolDerived",
    "DefectFlags",
    "MeltPoolModel",
]

_EPS = 1.0e-12


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PoolState:
    """The four melt-pool states.  SI units."""

    T_pool: float
    w: float
    p: float
    f: float

    def to_array(self) -> np.ndarray:
        return np.array([self.T_pool, self.w, self.p, self.f], dtype=float)

    @staticmethod
    def from_array(x: np.ndarray) -> "PoolState":
        return PoolState(float(x[0]), float(x[1]), float(x[2]), float(x[3]))

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PoolInputs:
    """Exogenous inputs to the pool over one step.  SI units."""

    I: float
    V: float
    v_travel: float
    v_wire: float
    gap: float
    thickness: float
    weave_amp: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.I,
                self.V,
                self.v_travel,
                self.v_wire,
                self.gap,
                self.thickness,
                self.weave_amp,
            ],
            dtype=float,
        )

    @staticmethod
    def from_array(u: np.ndarray) -> "PoolInputs":
        return PoolInputs(*(float(v) for v in u))


@dataclass(frozen=True, slots=True)
class PoolDerived:
    """Algebraic quantities computed on the way; useful for logging and tests."""

    A_fz: float
    A_ss: float
    l_eff: float
    volume: float
    AR: float
    AR_tgt: float
    tau_A: float
    q_arc: float
    q_cond: float
    q_conv: float
    q_rad: float
    q_net: float
    q_fus: float
    q_sh: float
    kappa_gap: float
    A_dep: float
    A_req: float
    f_ss: float
    mode: TransferMode
    superheat: float

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d


@dataclass(frozen=True, slots=True)
class DefectFlags:
    """Instantaneous defect indicators."""

    burn_through: bool
    burn_through_thickness: bool
    burn_through_bridging: bool
    lack_of_fusion: bool
    lof_cold: bool
    lof_sidewall: bool
    lof_underfill: bool
    w_root: float
    w_crit: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class MeltPoolModel:
    """Reduced-order melt-pool process model.

    The same object is used three ways: as the plant inside ``sim/cell.py``,
    as the process model inside ``estimation/ekf.py``, and as the predictor
    the controller uses to look ahead one horizon.  That is deliberate — the
    demo must be honest that the estimator's model *is* the simulator's model
    here, and that on real hardware they will differ (which is what the
    learned residual in ``estimation/residual.py`` is for).
    """

    def __init__(self, cfg: WeldConfig) -> None:
        self.cfg = cfg

    # -- helpers ---------------------------------------------------------
    def initial_state(self) -> PoolState:
        """A just-ignited pool: melting point, minimal dimensions, empty joint."""
        c = self.cfg
        return PoolState(
            T_pool=c.material.T_m,
            w=4.0 * c.pool.w_floor,
            p=4.0 * c.pool.p_floor,
            f=0.0,
        )

    def _clamp(self, s: PoolState) -> PoolState:
        c = self.cfg
        return PoolState(
            T_pool=clip(s.T_pool, c.material.T_m, 4000.0),
            w=clip(s.w, c.pool.w_floor, 0.10),
            p=clip(s.p, c.pool.p_floor, 0.10),
            f=clip(s.f, 0.0, c.pool.f_max),
        )

    # -- algebra ---------------------------------------------------------
    def derived(self, s: PoolState, u: PoolInputs) -> PoolDerived:
        """Evaluate every algebraic quantity for the current (state, input)."""
        c = self.cfg
        mat, pl, cons = c.material, c.pool, c.consumable

        w = max(s.w, pl.w_floor)
        p = max(s.p, pl.p_floor)
        v = max(u.v_travel, 1.0e-5)

        A_fz = math.pi / 4.0 * w * p
        l_eff = pl.kappa_L * w
        volume = A_fz * l_eff
        A_surf = math.pi / 4.0 * w * l_eff
        AR = w / p

        gap_ratio = min(max(u.gap, 0.0) / w, 1.0)
        kappa_gap = 1.0 - pl.k_gap_cond * gap_ratio

        dT = s.T_pool - mat.T_0
        q_arc = pl.eta_arc * u.V * u.I
        q_cond = pl.C_cond * mat.k * w * dT * kappa_gap
        q_conv = pl.h_conv * A_surf * dT
        q_rad = mat.emissivity * SIGMA_SB * A_surf * (s.T_pool**4 - mat.T_0**4)
        q_net = q_arc - q_cond - q_conv - q_rad

        q_fus = pl.eta_melt * max(q_net, 0.0)
        q_sh = (1.0 - pl.eta_melt) * q_net

        A_ss = q_fus / (mat.rho * mat.h_m * v)
        tau_A = clip(l_eff / v, pl.tau_A_min, pl.tau_A_max)

        mode = classify_transfer_mode(u.I, c.arc)
        AR_tgt = (
            pl.AR_0
            * (pl.I_ref / max(u.I, 1.0)) ** pl.n_I
            * (pl.v_ref / v) ** pl.n_v
            * pl.ar_mode_gain[mode]
            * (1.0 + pl.k_weave * 2.0 * max(u.weave_amp, 0.0) / w)
            / (1.0 + pl.k_gap_ar * gap_ratio)
        )
        AR_tgt = clip(AR_tgt, pl.AR_min, pl.AR_max)

        A_dep = cons.eta_dep * cons.area * max(u.v_wire, 0.0) / v
        A_req = max(u.gap, 0.0) * u.thickness + c.joint.A_reinf
        f_ss = clip(A_dep / max(A_req, _EPS), 0.0, pl.f_max)

        superheat = max(s.T_pool - mat.T_m, 0.0) / (mat.T_m - mat.T_0)

        return PoolDerived(
            A_fz=A_fz,
            A_ss=A_ss,
            l_eff=l_eff,
            volume=volume,
            AR=AR,
            AR_tgt=AR_tgt,
            tau_A=tau_A,
            q_arc=q_arc,
            q_cond=q_cond,
            q_conv=q_conv,
            q_rad=q_rad,
            q_net=q_net,
            q_fus=q_fus,
            q_sh=q_sh,
            kappa_gap=kappa_gap,
            A_dep=A_dep,
            A_req=A_req,
            f_ss=f_ss,
            mode=mode,
            superheat=superheat,
        )

    # -- dynamics --------------------------------------------------------
    def derivatives(
        self, s: PoolState, u: PoolInputs, d: PoolDerived | None = None
    ) -> tuple[np.ndarray, PoolDerived]:
        """Continuous-time derivative ``dx/dt`` of the four states."""
        c = self.cfg
        mat, pl, cons = c.material, c.pool, c.consumable
        d = self.derived(s, u) if d is None else d

        w = max(s.w, pl.w_floor)
        p = max(s.p, pl.p_floor)
        v = max(u.v_travel, 1.0e-5)

        # --- temperature -------------------------------------------------
        m_wire = mat.rho * cons.area * max(u.v_wire, 0.0)
        C_th = mat.rho * mat.c_p * d.volume
        G_th = mat.rho * mat.c_p * d.A_fz * v + m_wire * mat.c_p
        if G_th > _EPS:
            C_th = min(C_th, pl.tau_T_max * G_th)
        C_th = max(C_th, 1.0e-6)

        q_out_solid = mat.rho * mat.c_p * (s.T_pool - mat.T_m) * d.A_fz * v
        q_out_wire = m_wire * (mat.c_p * (s.T_pool - mat.T_0) + mat.L_f)
        dT = (d.q_sh - q_out_solid - q_out_wire) / C_th

        # --- fusion area and aspect ratio --------------------------------
        dA = (d.A_ss - d.A_fz) / d.tau_A
        dAR = (d.AR_tgt - d.AR) / pl.tau_ar

        # --- map (dA, dAR) -> (dw, dp) -----------------------------------
        S = 4.0 / math.pi * dA
        D = p * p * dAR
        dw = (S + D) / (2.0 * p)
        dp = (S - D) / (2.0 * w)

        # --- gap fill ----------------------------------------------------
        df = (d.f_ss - s.f) / pl.tau_f

        return np.array([dT, dw, dp, df], dtype=float), d

    def step(
        self, s: PoolState, u: PoolInputs, dt: float
    ) -> tuple[PoolState, PoolDerived]:
        """Advance one step (semi-implicit Euler) and clamp to physical bounds.

        Semi-implicit rather than RK4: every time constant in the model is
        >= 20 ms while the simulation step is 200 us, so the explicit error is
        negligible and the cost is 1 evaluation instead of 4.  That matters —
        the EKF differentiates this function numerically at 50 Hz.
        """
        dx, d = self.derivatives(s, u)
        nxt = PoolState(
            T_pool=s.T_pool + dt * float(dx[0]),
            w=s.w + dt * float(dx[1]),
            p=s.p + dt * float(dx[2]),
            f=s.f + dt * float(dx[3]),
        )
        return self._clamp(nxt), d

    def step_array(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Array-in / array-out wrapper, for the EKF's numerical Jacobian."""
        s, _ = self.step(PoolState.from_array(x), PoolInputs.from_array(u), dt)
        return s.to_array()

    # -- equilibrium -----------------------------------------------------
    def steady_state(
        self, u: PoolInputs, *, iters: int = 4000, dt: float = 2.0e-3
    ) -> PoolState:
        """Integrate to equilibrium for the given constant inputs.

        Used to initialise the simulator and the EKF, and by the controller as
        a cheap feed-forward map.  Marching the ODE is more robust than a
        root-find because of the clamps.
        """
        s = self.initial_state()
        for _ in range(iters):
            nxt, _ = self.step(s, u, dt)
            if (
                abs(nxt.p - s.p) < 1.0e-9
                and abs(nxt.w - s.w) < 1.0e-9
                and abs(nxt.T_pool - s.T_pool) < 1.0e-6
            ):
                return nxt
            s = nxt
        return s

    def effective_melting_efficiency(self, s: PoolState, u: PoolInputs) -> float:
        """Realised melting efficiency ``q_fus / q_arc`` [-].

        ``eta_melt`` in the config is applied to the *net* power, so the
        realised value is lower.  Reporting it keeps the comparison with the
        textbook relation in ``heat_source.fusion_area_from_heat_input``
        honest.
        """
        d = self.derived(s, u)
        return float(d.q_fus / d.q_arc) if d.q_arc > _EPS else 0.0

    # -- defects ---------------------------------------------------------
    def defects(self, s: PoolState, u: PoolInputs) -> DefectFlags:
        """Evaluate the burn-through and lack-of-fusion conditions."""
        c = self.cfg
        h = u.thickness
        w_crit = c.defect.c_st * c.material.capillary_length

        bt_thickness = s.p >= c.defect.beta_bt * h
        # The molten span at the root only exists once the pool nears the root.
        root_frac = clip((s.p - 0.6 * h) / (0.4 * h), 0.0, 1.0)
        w_root = root_frac * s.w + max(u.gap, 0.0)
        # Deposited filler physically supports the root, so it raises the limit
        # smoothly (a hard f < 1 gate would put a discontinuity in front of the
        # controller).
        w_crit *= 1.0 + c.defect.k_fill_support * min(max(s.f, 0.0), 1.0)
        bt_bridging = bool(w_root > w_crit)

        lof_cold = s.p < c.defect.p_min
        lof_sidewall = s.w < max(u.gap, 0.0) + 2.0 * c.joint.sidewall_margin
        lof_underfill = s.f < c.defect.f_min

        return DefectFlags(
            burn_through=bool(bt_thickness or bt_bridging),
            burn_through_thickness=bool(bt_thickness),
            burn_through_bridging=bt_bridging,
            lack_of_fusion=bool(lof_cold or lof_sidewall or lof_underfill),
            lof_cold=bool(lof_cold),
            lof_sidewall=bool(lof_sidewall),
            lof_underfill=bool(lof_underfill),
            w_root=float(w_root),
            w_crit=float(w_crit),
        )

    def with_state(self, s: PoolState, **updates: float) -> PoolState:
        """Return a copy of ``s`` with fields replaced (convenience for tests)."""
        return self._clamp(replace(s, **updates))
