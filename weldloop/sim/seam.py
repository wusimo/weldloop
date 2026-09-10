"""Seam geometry generator.

Produces the *ground-truth* root-gap profile ``g(s)`` and lateral misalignment
``o(s)`` along a butt joint of length ``L``.  This is the disturbance the whole
demo is built around: fit-up error makes the gap vary along the seam, and a
fixed-parameter weld cannot be right everywhere.

Profiles are deterministic given a seed.  Each is built as
``base shape -> Gaussian smoothing -> white fit-up roughness -> clip``, so the
result is smooth on the scale of the pool but rough on the scale of the
profiler.

TODO(real-hw): in the real phase-1 capture this class is replaced by the
measured profile from the laser profiler (or a CMM scan of the fit-up), read
from the same ``Seam`` container.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weldloop.config import SeamConfig

__all__ = ["Seam", "make_seam"]


def _gaussian_smooth(y: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Reflect-padded Gaussian smoothing (kept local to avoid a scipy import)."""
    if sigma_samples <= 0.5:
        return y
    half = int(np.ceil(3.0 * sigma_samples))
    t = np.arange(-half, half + 1, dtype=float)
    kern = np.exp(-0.5 * (t / sigma_samples) ** 2)
    kern /= kern.sum()
    padded = np.pad(y, half, mode="reflect")
    return np.convolve(padded, kern, mode="valid")


@dataclass(frozen=True)
class Seam:
    """A sampled seam.

    Attributes
    ----------
    s:      arc-length stations along the seam [m], uniform, step ``ds``
    gap:    root gap at each station [m]
    offset: lateral misalignment at each station [m]
    """

    s: np.ndarray
    gap: np.ndarray
    offset: np.ndarray
    kind: str
    seed: int

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def gap_at(self, s: float | np.ndarray) -> float | np.ndarray:
        """Root gap [m] at arc-length ``s`` (linear interpolation, clamped ends)."""
        return np.interp(s, self.s, self.gap)

    def offset_at(self, s: float | np.ndarray) -> float | np.ndarray:
        """Lateral misalignment [m] at arc-length ``s``."""
        return np.interp(s, self.s, self.offset)


def make_seam(cfg: SeamConfig, seed: int = 0) -> Seam:
    """Build a seam with the configured gap profile.

    ``kind`` is one of:

    ``constant``  flat gap at the midpoint of [gap_min, gap_max]
    ``step``      gap_min -> gap_max -> gap_min, two sharp fit-up steps.
                  This is the headline demo case: the controller has to react
                  to a step it cannot see coming from the arc alone.
    ``ramp``      linear open-up along the seam (tack-weld distortion)
    ``sine``      one and a half periods of gap breathing
    ``random``    smoothed random walk, clipped to the band
    """
    rng = np.random.default_rng(seed)
    n = max(int(round(cfg.length / cfg.ds)) + 1, 8)
    s = np.linspace(0.0, cfg.length, n)
    x = s / cfg.length
    lo, hi = cfg.gap_min, cfg.gap_max
    mid = 0.5 * (lo + hi)

    if cfg.kind == "constant":
        g = np.full(n, mid)
    elif cfg.kind == "step":
        g = np.full(n, lo)
        g[(x >= 0.35) & (x < 0.70)] = hi
        g[x >= 0.70] = mid
    elif cfg.kind == "ramp":
        g = lo + (hi - lo) * x
    elif cfg.kind == "sine":
        g = mid + 0.5 * (hi - lo) * np.sin(2.0 * np.pi * 1.5 * x)
    elif cfg.kind == "random":
        steps = rng.normal(0.0, 1.0, n).cumsum()
        steps -= steps.mean()
        span = np.ptp(steps)
        g = mid + (hi - lo) * 0.5 * (steps / (span if span > 0 else 1.0)) * 2.0
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown gap profile kind: {cfg.kind!r}")

    sigma = cfg.smooth_len / cfg.ds
    g = _gaussian_smooth(np.asarray(g, dtype=float), sigma)
    g = g + rng.normal(0.0, cfg.noise_std, n)
    g = np.clip(g, 0.0, None)

    off = _gaussian_smooth(rng.normal(0.0, 1.0, n), 4.0 * sigma)
    peak = float(np.max(np.abs(off)))
    off = off / (peak if peak > 0 else 1.0) * cfg.misalign_max

    return Seam(s=s, gap=g, offset=off, kind=cfg.kind, seed=seed)
