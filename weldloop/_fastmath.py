"""Small scalar helpers kept out of the hot path's way.

``numpy.clip`` on a Python float costs ~3 us of dispatch; at 5 kHz with a
dozen clips per step that dominates the simulation.  These are the scalar
equivalents.
"""

from __future__ import annotations

__all__ = ["clip", "clip_lo"]


def clip(x: float, lo: float, hi: float) -> float:
    """Scalar clamp to ``[lo, hi]``."""
    return lo if x < lo else (hi if x > hi else x)


def clip_lo(x: float, lo: float) -> float:
    """Scalar clamp from below."""
    return lo if x < lo else x
