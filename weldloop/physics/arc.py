"""Arc characteristic, metal transfer and short-circuit statistics.

This is the "power source as a sensor" half of the physics.  Three ideas
matter for the proposal:

1. **Static arc characteristic.**  ``V = V_0 + E_a * L_arc + R_so(I) * I``.
   Because the resistive term is known from the stickout and the wire, the
   arc *length* can be inferred from the measured V/I pair at kHz rate.  This
   is the classical through-the-arc sensing principle.

2. **Pool surface oscillation — the actual sensing mechanism.**  With the wire
   feed speed fixed, mass conservation (wire fed = wire melted) pins the mean
   current, and a constant-voltage machine pins the mean voltage.  So the *DC*
   levels of V and I say almost nothing about penetration.  This is a real
   property of CV GMAW, not a modelling artefact, and pretending otherwise
   would make the whole proposal dishonest.

   What does carry the information is that the pool surface oscillates under
   the arc.  Its frequency follows the classical surface-tension scaling for a
   liquid pool of radius R = w/2::

       f_osc = C_osc * sqrt(gamma / (rho * R^3))          -> pool WIDTH

   and its amplitude grows with pool depth and superheat::

       a_osc = k_a_osc * p * (1 + k_osc_T * superheat)     -> pool DEPTH

   The oscillation modulates the instantaneous arc length, hence the
   instantaneous voltage, at kHz-observable rates.  Band-power and
   ripple-frequency features of the 5 kHz V/I stream are therefore genuinely
   informative, while the means are not.  ``estimation/features.py`` extracts
   exactly those.

3. **Metal transfer / short circuits.**  A short circuit happens when the
   oscillating surface reaches the wire tip, so the short-circuit rate scales
   with the ripple-to-arc-length ratio ``a_osc / L_arc`` — a third,
   independent channel onto the same pool state.

All coefficients are lumped and CONFIGURABLE (``config.ArcConfig``); none is
quoted from a publication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from weldloop.config import ArcConfig, ConsumableConfig, MaterialConfig, TransferMode

__all__ = [
    "stickout_resistance",
    "arc_voltage",
    "arc_length_from_voltage",
    "classify_transfer_mode",
    "melting_rate",
    "current_for_melting_rate",
    "pool_oscillation",
    "mean_arc_length",
    "short_circuit_rate",
    "ShortCircuitProcess",
    "ArcSample",
    "ArcModel",
]


# --------------------------------------------------------------------------
# static characteristic
# --------------------------------------------------------------------------
def stickout_resistance(
    stickout: float, cons: ConsumableConfig, rho_scale: float = 1.0
) -> float:
    """Ohmic resistance of the electrode extension [ohm].

    ``R = rho_scale * rho_e * stickout / A_wire``.  This term is what makes the
    measured voltage depend on stickout, and ``rho_scale`` is the slow
    contact-tip-wear drift that the estimator does not know about — it makes a
    resistance drift look like an arc-length change to a naive estimator.
    """
    return rho_scale * cons.rho_e * max(stickout, 0.0) / cons.area


def arc_voltage(
    I: float,
    L_arc: float,
    stickout: float,
    cfg: ArcConfig,
    cons: ConsumableConfig,
    rho_scale: float = 1.0,
) -> float:
    """Static arc characteristic: terminal voltage [V]."""
    return (
        cfg.V_0
        + cfg.E_a * max(L_arc, 0.0)
        + stickout_resistance(stickout, cons, rho_scale) * I
    )


def arc_length_from_voltage(
    V: float, I: float, stickout: float, cfg: ArcConfig, cons: ConsumableConfig
) -> float:
    """Invert the arc characteristic to an arc length [m].

    This is the estimator's view: it uses the *nominal* stickout, so any real
    stickout drift shows up as an arc-length error.  That bias is exactly why
    the geometric and thermal sensors are still needed.
    """
    L = (V - cfg.V_0 - stickout_resistance(stickout, cons) * I) / cfg.E_a
    return float(max(L, 0.0))


def classify_transfer_mode(I: float, cfg: ArcConfig) -> TransferMode:
    """Droplet-transfer mode from the mean current."""
    if I < cfg.I_globular:
        return TransferMode.SHORT_CIRCUIT
    if I < cfg.I_spray:
        return TransferMode.GLOBULAR
    return TransferMode.SPRAY


# --------------------------------------------------------------------------
# burn-off / self-regulation
# --------------------------------------------------------------------------
def melting_rate(I: float, stickout: float, cfg: ArcConfig) -> float:
    """Wire melting rate [m/s] from the GMAW burn-off law.

    ``v_wire = a * I + b * stickout * I^2``: the linear term is arc heating at
    the wire tip, the quadratic term is resistive (I^2 R) heating of the
    electrode extension.  In a constant-voltage machine this law, together
    with the arc characteristic, produces the self-regulation that keeps the
    arc length roughly constant — which is what the inner loop models.
    """
    return cfg.a_burn * I + cfg.b_burn * max(stickout, 0.0) * I * I


def current_for_melting_rate(v_wire: float, stickout: float, cfg: ArcConfig) -> float:
    """Invert the burn-off law: current [A] that melts wire at ``v_wire``."""
    a, b = cfg.a_burn, cfg.b_burn * max(stickout, 0.0)
    if b <= 0.0:
        return max(v_wire / a, 0.0)
    disc = a * a + 4.0 * b * max(v_wire, 0.0)
    return float((-a + math.sqrt(disc)) / (2.0 * b))


# --------------------------------------------------------------------------
# pool oscillation, arc-length coupling, short-circuit statistics
# --------------------------------------------------------------------------
def pool_oscillation(
    w: float, p: float, superheat: float, cfg: ArcConfig, mat: MaterialConfig
) -> tuple[float, float]:
    """Pool-surface oscillation frequency [Hz] and amplitude [m].

    ``f_osc = C_osc * sqrt(gamma / (rho * (w/2)^3))`` is the surface-tension
    (Rayleigh-type) scaling for a liquid pool of radius ``w/2``: wider pool,
    lower frequency.  ``C_osc`` is a single lumped constant — CONFIGURABLE,
    and the first thing to identify from real V/I spectra.

    ``a_osc = k_a_osc * p * (1 + k_osc_T * superheat)``: a deeper, hotter pool
    has more liquid to slosh and a weaker restoring surface, so it ripples
    harder.
    """
    R = max(0.5 * w, 1.0e-4)
    f_osc = cfg.C_osc * math.sqrt(mat.gamma / (mat.rho * R**3))
    f_osc = float(min(f_osc, cfg.f_osc_max))
    a_osc = cfg.k_a_osc * max(p, 0.0) * (1.0 + cfg.k_osc_T * max(superheat, 0.0))
    return f_osc, float(a_osc)


def mean_arc_length(L_geom: float, penetration: float, cfg: ArcConfig) -> float:
    """Mean arc length [m]: geometric standoff plus the pool surface depression.

    ``L = (CTWD - stickout) + k_sag * p``.  The depression is *not* directly
    observable — a constant-voltage machine absorbs it into the electrode
    extension — which is exactly why it shows up as a slow bias on any naive
    arc-length-based penetration estimate.
    """
    return max(L_geom + cfg.k_sag * max(penetration, 0.0), 1.0e-4)


def short_circuit_rate(
    I: float, L_arc: float, a_osc: float, cfg: ArcConfig
) -> float:
    """Expected short-circuit frequency [Hz].

    ``f_sc = f_max * exp(-((I - I_peak)/sigma)^2) * exp(-k * L_arc / a_osc)``

    Bell-shaped in current (no shorts in spray transfer, none in a cold
    stubbing arc) and controlled by how far the surface has to travel to reach
    the wire tip relative to how hard it is rippling.
    """
    bell = math.exp(-(((I - cfg.I_sc_peak) / cfg.sigma_I_sc) ** 2))
    ratio = max(L_arc, 1.0e-5) / max(a_osc, 1.0e-6)
    return float(max(cfg.f_sc_max * bell * math.exp(-cfg.k_sc_ratio * ratio), 0.0))


class ShortCircuitProcess:
    """Poisson short-circuit event generator with finite event duration.

    Stepped at the simulation rate.  Deterministic given the ``Generator``.
    """

    def __init__(self, cfg: ArcConfig, rng: np.random.Generator) -> None:
        self._cfg = cfg
        self._rng = rng
        self.in_short = False
        self._t_left = 0.0
        self.n_events = 0

    def step(self, dt: float, rate_hz: float) -> bool:
        """Advance by ``dt`` and return True while a short circuit is active."""
        if self.in_short:
            self._t_left -= dt
            if self._t_left <= 0.0:
                self.in_short = False
            return self.in_short
        if rate_hz > 0.0 and self._rng.random() < rate_hz * dt:
            self.in_short = True
            self.n_events += 1
            self._t_left = (
                float(self._rng.exponential(self._cfg.t_short))
                + 0.25 * self._cfg.t_short
            )
        return self.in_short


@dataclass(frozen=True, slots=True)
class ArcSample:
    """One instantaneous electrical sample from the arc."""

    V: float
    I: float
    in_short: bool
    L_inst: float
    L_mean: float
    f_osc: float
    a_osc: float
    f_sc: float
    mode: TransferMode


class ArcModel:
    """Arc characteristic + pool oscillation + short-circuit event process.

    Holds the oscillation phase, so it must be stepped at the simulation rate.
    """

    def __init__(
        self, cfg: ArcConfig, cons: ConsumableConfig, mat: MaterialConfig,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.cons = cons
        self.mat = mat
        self.shorts = ShortCircuitProcess(cfg, rng)
        self._rng = rng
        self.phase = 0.0
        self.f_osc = 0.0
        self.a_osc = 0.0

    def advance(
        self, dt: float, *, L_geom: float, w: float, p: float, superheat: float
    ) -> tuple[float, float]:
        """Advance the oscillation phase; return (instantaneous, mean) arc length.

        Called *before* the power source steps, so the machine's current loop
        sees the same instantaneous arc length the arc does.
        """
        self.f_osc, self.a_osc = pool_oscillation(w, p, superheat, self.cfg, self.mat)
        self.phase = (self.phase + 2.0 * math.pi * self.f_osc * dt) % (2.0 * math.pi)
        L_mean = mean_arc_length(L_geom, p, self.cfg)
        L_inst = max(L_mean + self.a_osc * math.sin(self.phase), 1.0e-5)
        return L_inst, L_mean

    def emit(
        self,
        dt: float,
        *,
        I: float,
        L_inst: float,
        L_mean: float,
        stickout: float,
        superheat: float,
        rho_scale: float = 1.0,
    ) -> ArcSample:
        """Produce the instantaneous V/I sample for the current arc state.

        During a short circuit the arc is extinguished: the voltage collapses
        to ``V_short`` and the current surges by ``I_short_gain``.  That surge
        is what a real inverter regulates, and it is why the raw 5 kHz waveform
        looks nothing like its own mean.
        """
        rate = short_circuit_rate(I, L_mean, self.a_osc, self.cfg)
        jitter = 1.0 + self.cfg.k_instab * max(superheat, 0.0)
        in_short = self.shorts.step(dt, rate * jitter)
        if in_short:
            V = self.cfg.V_short
            I_out = I * self.cfg.I_short_gain
        else:
            V = arc_voltage(I, L_inst, stickout, self.cfg, self.cons, rho_scale)
            I_out = I
        return ArcSample(
            V=float(V),
            I=float(I_out),
            in_short=in_short,
            L_inst=float(L_inst),
            L_mean=float(L_mean),
            f_osc=float(self.f_osc),
            a_osc=float(self.a_osc),
            f_sc=float(rate),
            mode=classify_transfer_mode(I, self.cfg),
        )
