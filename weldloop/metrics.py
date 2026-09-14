"""Scoring one weld.

These are the numbers the proposal is argued with, so each one is defined here
once and computed the same way for every controller.  All of them are computed
from ``truth_`` columns — they are the *quality of the weld that was actually
made*, not the quality of anyone's estimate of it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["WeldMetrics", "score"]


@dataclass(frozen=True, slots=True)
class WeldMetrics:
    """Everything worth reporting about one weld."""

    controller: str
    duration_s: float
    seam_length_mm: float
    mean_travel_speed_mm_s: float

    burn_through_events: int  # runs longer than ``defect.min_hole_length``
    burn_through_length_mm: float
    lack_of_fusion_length_mm: float
    lof_cold_length_mm: float
    lof_sidewall_length_mm: float
    lof_underfill_length_mm: float

    penetration_mean_mm: float
    penetration_std_mm: float
    penetration_in_band_pct: float
    fill_mean: float
    fill_min: float

    defect_free_length_pct: float

    def as_dict(self) -> dict:
        return asdict(self)

    def headline(self) -> str:
        return (
            f"{self.controller:<10s} "
            f"BT {self.burn_through_events:>2d} ({self.burn_through_length_mm:5.1f} mm)  "
            f"LOF {self.lack_of_fusion_length_mm:6.1f} mm  "
            f"p {self.penetration_mean_mm:.2f}+-{self.penetration_std_mm:.2f} mm  "
            f"in-band {self.penetration_in_band_pct:5.1f} %  "
            f"v {self.mean_travel_speed_mm_s:.2f} mm/s  "
            f"cycle {self.duration_s:5.1f} s"
        )


def _run_length(flag: np.ndarray, ds: np.ndarray) -> float:
    return float(np.sum(ds[flag]))


def _events(flag: np.ndarray, ds: np.ndarray, min_length: float) -> int:
    """Count burn-through *holes*, not sample-level flag flickers.

    A run of the flag shorter than ``min_length`` of seam is a marginal
    excursion of the bridging criterion, not a hole in the plate.  Counting
    those would make the metric depend on the sample rate, which is exactly
    the kind of number that falls apart under questioning.
    """
    f = flag.astype(np.int8)
    if not f.any():
        return 0
    edges = np.flatnonzero(np.diff(f))
    starts = list(edges[f[edges + 1] == 1] + 1)
    ends = list(edges[f[edges + 1] == 0] + 1)
    if f[0]:
        starts = [0] + starts
    if f[-1]:
        ends = ends + [len(f)]
    return int(sum(1 for a, b in zip(starts, ends) if np.sum(ds[a:b]) >= min_length))


def score(table, cfg, name: str) -> WeldMetrics:
    """Score a finished weld from its log.

    ``lack of fusion`` is reported both in total and split by cause, because
    the three causes call for different fixes: too cold, pool not reaching the
    sidewalls, or simply not enough filler for the gap.
    """
    s = table["rb_s"]
    ds = np.diff(s, prepend=s[0])
    p = table["truth_penetration"]
    f = table["truth_fill"]
    gap = table["truth_gap"]
    w = table["truth_pool_w"]

    bt = table["truth_burn_through"] > 0.5
    lof = table["truth_lack_of_fusion"] > 0.5
    lof_cold = p < cfg.defect.p_min
    lof_side = w < gap + 2.0 * cfg.joint.sidewall_margin
    lof_fill = f < cfg.defect.f_min

    # The acceptance band is a *fraction of plate thickness*, which only shows
    # up as a constant when the plate is.  On a stepped joint, scoring the thin
    # section against the thick section's band would call a good weld bad and
    # a burnt-through one fine.
    h = table["truth_thickness"]
    frac_lo = cfg.control.p_lo / cfg.joint.thickness
    frac_hi = cfg.control.p_hi / cfg.joint.thickness
    in_band = (p >= frac_lo * h) & (p <= frac_hi * h)
    clean = ~bt & ~lof
    length = float(s[-1] - s[0])

    return WeldMetrics(
        controller=name,
        duration_s=float(table["t"][-1] - table["t"][0]),
        seam_length_mm=length * 1e3,
        mean_travel_speed_mm_s=float(np.mean(table["rb_v_travel"]) * 1e3),
        burn_through_events=_events(bt, ds, cfg.defect.min_hole_length),
        burn_through_length_mm=_run_length(bt, ds) * 1e3,
        lack_of_fusion_length_mm=_run_length(lof, ds) * 1e3,
        lof_cold_length_mm=_run_length(lof_cold, ds) * 1e3,
        lof_sidewall_length_mm=_run_length(lof_side, ds) * 1e3,
        lof_underfill_length_mm=_run_length(lof_fill, ds) * 1e3,
        penetration_mean_mm=float(np.mean(p) * 1e3),
        penetration_std_mm=float(np.std(p) * 1e3),
        penetration_in_band_pct=float(100.0 * np.mean(in_band)),
        fill_mean=float(np.mean(f)),
        fill_min=float(np.min(f)),
        defect_free_length_pct=float(100.0 * _run_length(clean, ds) / max(length, 1e-9)),
    )
