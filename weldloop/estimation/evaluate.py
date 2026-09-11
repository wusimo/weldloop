"""Offline (but strictly causal) evaluation of the estimator over a logged weld.

Why offline: the V/I features are identical for every sensor ablation, so they
are extracted once and the four filters are run over the same feature series.
That makes the ablation a fair comparison — same weld, same noise realisation,
same features, only the fusion differs — and it keeps the whole Phase 3 table
inside a few seconds.

Nothing here is acausal.  Each filter step sees only samples with a timestamp
at or before it, which is why the identical code drives the controller online
in Phase 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from weldloop.config import WeldConfig
from weldloop.estimation.ekf import SENSOR_SETS, PoolEKF, SensorSet
from weldloop.estimation.features import FeatureStream, GapTracker

__all__ = ["Series", "EstimationResult", "extract_series", "run_ekf", "run_all"]


@dataclass(slots=True)
class Series:
    """Everything the filter needs, resampled to the filter rate."""

    t: np.ndarray
    s: np.ndarray
    I: np.ndarray
    V: np.ndarray
    v_travel: np.ndarray
    v_wire: np.ndarray
    weave: np.ndarray
    gap_profiler: np.ndarray
    gap_assumed: np.ndarray
    f_ripple: np.ndarray
    a_ripple: np.ndarray
    f_sc: np.ndarray
    L_arc_est: np.ndarray
    ir_T: np.ndarray
    ir_w: np.ndarray
    rgb_w: np.ndarray
    truth_p: np.ndarray
    truth_w: np.ndarray
    truth_T: np.ndarray
    truth_f: np.ndarray
    truth_gap: np.ndarray

    def __len__(self) -> int:
        return len(self.t)


def _fresh_values(col: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Latest non-NaN value of ``col`` that arrived inside each filter interval.

    NaN where nothing new arrived — so a 30 Hz camera against a 50 Hz filter
    contributes to some steps and not others, which is the honest behaviour.
    """
    have = np.flatnonzero(~np.isnan(col))
    out = np.full(len(rows), np.nan)
    if have.size == 0:
        return out
    prev = np.concatenate([[-1], rows[:-1]])
    lo = np.searchsorted(have, prev, side="right")
    hi = np.searchsorted(have, rows, side="right")
    fresh = hi > lo
    out[fresh] = col[have[hi[fresh] - 1]]
    return out


def extract_series(cfg: WeldConfig, table, assumed_gap: float = 0.0) -> Series:
    """Run the V/I feature extractor and resample every channel to the filter rate.

    ``assumed_gap`` is what a cell with no seam profiler has to fall back on:
    the gap the welding procedure specifies.  For a square butt that is zero,
    and the "V/I only" row of the ablation is exactly the cost of that
    assumption being wrong.
    """
    f_master = table.f_master
    dt_f = cfg.ekf.dt
    step = int(round(dt_f * f_master))
    n_win = int(round(cfg.estimator_window * f_master))

    V, I, sh = table["ps_V"], table["ps_I"], table["ps_short"]
    rows = np.arange(n_win, table.n_rows, step)

    stream = FeatureStream(cfg)
    tracker = GapTracker(default_gap=assumed_gap)

    n = len(rows)
    f_rip = np.empty(n); a_rip = np.empty(n); f_sc = np.empty(n)
    L_est = np.empty(n); I_m = np.empty(n); V_m = np.empty(n)
    gap_prof = np.empty(n)

    prof_gap, prof_lead, prof_valid = (
        table["prof_gap"], table["prof_lead_s"], table["prof_valid"]
    )
    s_col = table["rb_s"]

    cursor = 0
    for k, row in enumerate(rows):
        # feed everything that arrived since the previous filter step
        for i in range(cursor, row):
            stream.push(V[i], I[i], sh[i])
            if prof_valid[i] == 1.0 and not math.isnan(prof_gap[i]):
                tracker.push(float(prof_lead[i]), float(prof_gap[i]))
        cursor = row

        feat = stream.extract(table["t"][row - 1])
        f_rip[k] = feat.f_ripple
        a_rip[k] = feat.a_ripple
        f_sc[k] = feat.f_sc
        L_est[k] = feat.L_arc_est
        I_m[k] = feat.I_mean
        V_m[k] = feat.V_mean
        gap_prof[k] = tracker.gap_at(float(s_col[row - 1]))

    idx = rows - 1
    return Series(
        t=table["t"][idx],
        s=s_col[idx],
        I=I_m,
        V=V_m,
        v_travel=table["rb_v_travel"][idx],
        v_wire=table["ps_v_wire"][idx],
        weave=table["cmd_weave_amp"][idx],
        gap_profiler=gap_prof,
        gap_assumed=np.full(n, assumed_gap),
        f_ripple=f_rip,
        a_ripple=a_rip,
        f_sc=f_sc,
        L_arc_est=L_est,
        ir_T=_fresh_values(table["ir_T_peak"], rows),
        ir_w=_fresh_values(table["ir_pool_width"], rows),
        rgb_w=_fresh_values(table["rgb_pool_width"], rows),
        truth_p=table["truth_penetration"][idx],
        truth_w=table["truth_pool_w"][idx],
        truth_T=table["truth_T_pool"][idx],
        truth_f=table["truth_fill"][idx],
        truth_gap=table["truth_gap"][idx],
    )


@dataclass(slots=True)
class EstimationResult:
    """One ablation's trajectory and its scores."""

    name: str
    t: np.ndarray
    p: np.ndarray
    p_std: np.ndarray
    w: np.ndarray
    w_std: np.ndarray
    T: np.ndarray
    fill: np.ndarray
    truth_p: np.ndarray
    truth_w: np.ndarray

    @property
    def rmse_p(self) -> float:
        return float(np.sqrt(np.mean((self.p - self.truth_p) ** 2)))

    @property
    def bias_p(self) -> float:
        return float(np.mean(self.p - self.truth_p))

    @property
    def rmse_w(self) -> float:
        return float(np.sqrt(np.mean((self.w - self.truth_w) ** 2)))

    @property
    def mean_p_std(self) -> float:
        return float(np.mean(self.p_std))

    @property
    def coverage_2sigma(self) -> float:
        """Fraction of the time the truth lies inside the +/-2 sigma band.

        A filter whose covariance is honest lands near 0.95.  Reporting this
        matters because the controller *consumes* the covariance: an
        overconfident filter would make the controller aggressive exactly when
        it should be careful.
        """
        return float(
            np.mean(np.abs(self.p - self.truth_p) <= 2.0 * np.maximum(self.p_std, 1e-12))
        )

    def summary_row(self) -> dict[str, float | str]:
        return {
            "sensor set": self.name,
            "RMSE p [mm]": round(self.rmse_p * 1e3, 3),
            "bias p [mm]": round(self.bias_p * 1e3, 3),
            "RMSE w [mm]": round(self.rmse_w * 1e3, 3),
            "mean sigma_p [mm]": round(self.mean_p_std * 1e3, 3),
            "2-sigma coverage": round(self.coverage_2sigma, 3),
        }


def run_ekf(
    cfg: WeldConfig,
    series: Series,
    sensor_set: SensorSet | str,
    *,
    residual=None,
    label: str | None = None,
) -> EstimationResult:
    """Run one filter over a pre-extracted series."""
    ss = sensor_set if isinstance(sensor_set, SensorSet) else SensorSet.named(sensor_set)
    ekf = PoolEKF(cfg, ss, residual=residual)
    gap = series.gap_profiler if ss.use_profiler else series.gap_assumed

    n = len(series)
    p = np.empty(n); p_std = np.empty(n); w = np.empty(n)
    w_std = np.empty(n); T = np.empty(n); fill = np.empty(n)

    thickness = cfg.joint.thickness
    for k in range(n):
        u = np.array(
            [
                series.I[k] if np.isfinite(series.I[k]) else cfg.baseline.I_set,
                series.V[k] if np.isfinite(series.V[k]) else 27.0,
                series.v_travel[k],
                series.v_wire[k],
                gap[k],
                thickness,
                series.weave[k],
            ]
        )
        meas = {
            "f_ripple": series.f_ripple[k],
            "a_ripple": series.a_ripple[k],
            "f_sc": series.f_sc[k],
            "ir_T_peak": series.ir_T[k],
            "ir_pool_width": series.ir_w[k],
            "rgb_pool_width": series.rgb_w[k],
            "L_arc_est": series.L_arc_est[k],
        }
        out = ekf.step(series.t[k], u, meas)
        p[k] = out.penetration
        p_std[k] = out.penetration_std
        w[k] = out.pool_width
        w_std[k] = out.pool_width_std
        T[k] = out.T_pool
        fill[k] = out.fill

    return EstimationResult(
        name=label or ss.name,
        t=series.t, p=p, p_std=p_std, w=w, w_std=w_std, T=T, fill=fill,
        truth_p=series.truth_p, truth_w=series.truth_w,
    )


def run_all(
    cfg: WeldConfig, table, *, residual=None, names: tuple[str, ...] | None = None
) -> dict[str, EstimationResult]:
    """The Phase 3 ablation: every sensor set over the same weld."""
    series = extract_series(cfg, table)
    names = names or ("vi", "vi+profiler", "all", "rgb")
    out = {name: run_ekf(cfg, series, name) for name in names}
    if residual is not None:
        out["all + residual"] = run_ekf(
            cfg, series, "all", residual=residual, label="all + residual"
        )
    return out
