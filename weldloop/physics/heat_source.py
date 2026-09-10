"""Analytic welding heat-source models.

Two classical models are provided.  They are used here as **structural
references and sanity anchors** for the reduced-order melt-pool model in
``melt_pool.py`` — the real-time loop does not evaluate them.

* Rosenthal's moving point-source solution (Rosenthal, 1946) — the
  quasi-steady analytic temperature field of a point source travelling over a
  semi-infinite body.  Gives closed-form isotherms, hence a first-principles
  estimate of fusion-zone width and depth for a given heat input.
* Goldak's double-ellipsoid volumetric source (Goldak, 1984) — the standard
  distributed source used in welding FE analysis.  Provided here so the demo
  can show a physically shaped power density and so a future FE-calibrated
  version of the ROM has the same interface.

Cited **by name only**: no parameter value in this repository is taken from
those papers.  All ellipsoid semi-axes and efficiencies are configurable and
must be identified from real data.

Units are SI throughout: [m], [s], [K], [W], [W/m^3].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from weldloop.config import MaterialConfig

__all__ = [
    "GoldakParams",
    "goldak_flux",
    "rosenthal_thick_plate",
    "rosenthal_trailing_centerline",
    "melt_isotherm_halfwidth",
    "melt_isotherm_depth",
    "heat_input_per_length",
    "fusion_area_from_heat_input",
]


# --------------------------------------------------------------------------
# Goldak double ellipsoid
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GoldakParams:
    """Semi-axes of Goldak's double ellipsoid.

    ``a_f``/``a_r`` are the front/rear semi-axes along the travel direction,
    ``b`` the half-width, ``c`` the depth.  ``f_f + f_r == 2`` by construction
    (the two half-ellipsoids each carry half of a normalised Gaussian).

    CONFIGURABLE — the usual engineering starting point is a_f ~ b/2,
    a_r ~ 2*b, b ~ half the pool width, c ~ the penetration; those are
    *rules of thumb*, not published values.
    """

    a_f: float = 3.0e-3
    a_r: float = 12.0e-3
    b: float = 7.0e-3
    c: float = 4.0e-3
    f_f: float = 0.6

    @property
    def f_r(self) -> float:
        return 2.0 - self.f_f


def goldak_flux(
    xi: np.ndarray | float,
    y: np.ndarray | float,
    z: np.ndarray | float,
    Q: float,
    g: GoldakParams,
) -> np.ndarray:
    """Volumetric power density [W/m^3] of Goldak's double ellipsoid.

    Parameters
    ----------
    xi, y, z:
        Coordinates in the **torch-fixed frame** [m].  ``xi > 0`` is ahead of
        the arc, ``z >= 0`` is into the plate.
    Q:
        Total power absorbed by the workpiece [W] (i.e. ``eta_arc * V * I``).
    g:
        Ellipsoid semi-axes.

    Notes
    -----
    q(xi,y,z) = 6*sqrt(3)*f*Q / (a*b*c*pi^{3/2}) *
                exp(-3 xi^2/a^2 - 3 y^2/b^2 - 3 z^2/c^2)
    with ``a = a_f, f = f_f`` ahead of the arc and ``a = a_r, f = f_r`` behind.
    Integrating over the half-space returns ``Q``.
    """
    xi = np.asarray(xi, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    a = np.where(xi >= 0.0, g.a_f, g.a_r)
    f = np.where(xi >= 0.0, g.f_f, g.f_r)

    pref = 6.0 * math.sqrt(3.0) * f * Q / (a * g.b * g.c * math.pi**1.5)
    expo = -3.0 * (xi**2 / a**2 + y**2 / g.b**2 + z**2 / g.c**2)
    # z < 0 is outside the workpiece
    return np.where(z >= 0.0, pref * np.exp(expo), 0.0)


# --------------------------------------------------------------------------
# Rosenthal moving point source, semi-infinite body ("thick plate")
# --------------------------------------------------------------------------
def rosenthal_thick_plate(
    xi: np.ndarray | float,
    y: np.ndarray | float,
    z: np.ndarray | float,
    Q: float,
    v: float,
    mat: MaterialConfig,
) -> np.ndarray:
    """Quasi-steady temperature field [K] of a moving point source.

    T = T_0 + Q / (2 pi k R) * exp( -v (xi + R) / (2 alpha) ),
    R = sqrt(xi^2 + y^2 + z^2)

    ``xi`` is measured in the torch frame, positive ahead of the arc.
    The singularity at R -> 0 is clipped at one micrometre.
    """
    xi = np.asarray(xi, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    R = np.sqrt(xi**2 + y**2 + z**2)
    R = np.maximum(R, 1.0e-6)
    return mat.T_0 + Q / (2.0 * math.pi * mat.k * R) * np.exp(
        -v * (xi + R) / (2.0 * mat.alpha)
    )


def rosenthal_trailing_centerline(
    distance_behind: np.ndarray | float, Q: float, mat: MaterialConfig
) -> np.ndarray:
    """Temperature [K] directly behind the arc on the surface centreline.

    On that ray ``xi = -d``, ``R = d``, so the exponential is unity and
    ``T = T_0 + Q / (2 pi k d)`` — independent of travel speed.  This is the
    classical result and is used as a regression anchor in the tests.
    """
    d = np.maximum(np.asarray(distance_behind, dtype=float), 1.0e-6)
    return mat.T_0 + Q / (2.0 * math.pi * mat.k * d)


def _max_T_at(y: float, z: float, Q: float, v: float, mat: MaterialConfig) -> float:
    """Maximum over ``xi`` of the Rosenthal field at a fixed (y, z)."""
    scale = max(2.0 * mat.alpha / max(v, 1e-6), 1.0e-4)

    def neg_T(xi: float) -> float:
        return -float(rosenthal_thick_plate(xi, y, z, Q, v, mat))

    res = minimize_scalar(neg_T, bounds=(-20.0 * scale, 0.5 * scale), method="bounded")
    return -float(res.fun)


def _bracketed_isotherm(
    f, lo: float, hi: float, expand: int = 40
) -> float:
    """Find where ``f`` crosses zero on ``[lo, hi]``, shrinking ``hi`` if needed."""
    if f(lo) < 0.0:
        return 0.0
    for _ in range(expand):
        if f(hi) < 0.0:
            return float(brentq(f, lo, hi, xtol=1e-8, rtol=1e-10))
        hi *= 1.6
        if hi > 1.0:  # 1 m — physically absurd, give up
            break
    return float(hi)


def melt_isotherm_halfwidth(Q: float, v: float, mat: MaterialConfig) -> float:
    """Half-width [m] of the ``T = T_m`` isotherm on the plate surface.

    Analytic anchor for the reduced-order pool width.
    """
    return _bracketed_isotherm(
        lambda y: _max_T_at(y, 0.0, Q, v, mat) - mat.T_m, 1.0e-6, 5.0e-3
    )


def melt_isotherm_depth(Q: float, v: float, mat: MaterialConfig) -> float:
    """Depth [m] of the ``T = T_m`` isotherm below the arc centreline.

    Analytic anchor for the reduced-order penetration.
    """
    return _bracketed_isotherm(
        lambda z: _max_T_at(0.0, z, Q, v, mat) - mat.T_m, 1.0e-6, 5.0e-3
    )


# --------------------------------------------------------------------------
# heat-input book-keeping
# --------------------------------------------------------------------------
def heat_input_per_length(V: float, I: float, v: float, eta_arc: float) -> float:
    """Arc energy delivered per unit weld length [J/m]: ``eta * V * I / v``."""
    return eta_arc * V * I / max(v, 1.0e-9)


def fusion_area_from_heat_input(
    V: float, I: float, v: float, eta_arc: float, eta_melt: float, mat: MaterialConfig
) -> float:
    """Steady-state fused cross-sectional area [m^2].

    ``A = eta_arc * eta_melt * V * I / (rho * h_m * v)``

    This is the textbook melting-efficiency relation.  The reduced-order ODE
    in ``melt_pool.py`` is built so that its equilibrium reproduces exactly
    this expression once conduction/convection/radiation losses have been
    folded into ``eta_melt``; ``test_physics.py`` checks that.
    """
    return heat_input_per_length(V, I, v, eta_arc) * eta_melt / (mat.rho * mat.h_m)
