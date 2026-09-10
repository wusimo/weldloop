#!/usr/bin/env python3
"""Phase 2 figure: the sensor suite and the time-aligned log.

Six panels:

1. **Sampling raster** — every sensor on one master clock, with dropouts
   marked.  This is the "one clock, many rates" claim, drawn.
2. **Column fill rate** — how often each column of the wide table is non-NaN.
   Literally the capture schema, sorted by rate.
3. **Profiler preview** — the laser scanner measures ahead of the arc, so the
   controller gets the gap before it arrives.
4. **IR vs RGB pool width** — the headline negative result.
5. **RGB quality vs smoke** — why panel 4 looks like that.
6. **Acoustics** — mic RMS and re-ignition clicks against the electrical
   short-circuit flag.

Usage::

    python scripts/plot_sensors.py --seed 0 --out out/phase2_sensors.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import matplotlib.pyplot as plt
import numpy as np

from weldloop.config import default_config
from weldloop.sim.logger import SCHEMA
from weldloop.sim.runner import simulate
from weldloop.viz.style import C, downsample, use_style

SENSOR_ROWS = [
    ("power source  5 kHz", "ps_V", None, C.adaptive),
    ("torch force  1 kHz", "force_N", None, C.aux),
    ("arc mic  20 kHz", "mic_p", None, C.aux2),
    ("laser profiler  30 Hz", "prof_gap", "prof_valid", C.band_ok),
    ("IR camera  30 Hz", "ir_T_peak", "ir_valid", C.warn),
    ("RGB camera  30 Hz", "rgb_quality", "rgb_valid", C.baseline),
]


def panel_raster(ax, T) -> None:
    t = T["t"]
    t0, span = 12.0, 0.30  # 300 ms mid-weld, where the gap is opening
    w = (t >= t0) & (t < t0 + span)
    tt = (t[w] - t0) * 1e3
    n_bad = 0
    for row, (label, col, valid_col, color) in enumerate(SENSOR_ROWS):
        got = ~np.isnan(T[col][w])
        bad = (T[valid_col][w] == 0.0) if valid_col is not None else np.zeros_like(got)
        good = got & ~bad
        ax.plot(tt[good], np.full(good.sum(), row), "|", color=color, ms=10, mew=1.2)
        if bad.any():
            n_bad += int(bad.sum())
            ax.plot(tt[bad], np.full(bad.sum(), row), "|", color=C.bad, ms=10, mew=2.4)
    ax.plot([], [], "|", color=C.bad, ms=9, mew=2.4,
            label=f"丢帧 / 被拒绝的帧（本窗口 {n_bad} 次）")
    ax.set_yticks(range(len(SENSOR_ROWS)))
    ax.set_yticklabels([r[0] for r in SENSOR_ROWS])
    ax.set_ylim(-0.6, len(SENSOR_ROWS) - 0.4)
    ax.set_xlabel(f"master clock [ms]  (window from t = {t0:.0f} s)")
    ax.set_title(f"同一主时钟上的采样栅格（{span * 1e3:.0f} ms 窗口）")
    ax.legend(loc="upper right")
    ax.grid(axis="x")


def panel_fill(ax, T) -> None:
    """Fill rate, expressed as the effective sample rate it implies.

    Plotting the raw NaN fraction squashes every 30 Hz column against zero; the
    implied rate on a log axis says the same thing and reads as the schema.
    """
    rows = []
    for spec in SCHEMA:
        if spec.name == "t":
            continue
        arr = T[spec.name]
        frac = np.count_nonzero(~np.isnan(arr)) / len(arr)
        rows.append((spec.name, max(frac * T.f_master, 0.05), spec.real_hw))
    rows.sort(key=lambda r: (r[1], r[0]))
    names = [r[0] for r in rows]
    rates = np.array([r[1] for r in rows])
    colors = [C.adaptive if r[2] else C.muted for r in rows]
    y = np.arange(len(names))
    ax.barh(y, rates, color=colors, height=0.74, log=True)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=5.8)
    for rate, ref, lab in (
        (30.0, C.band_ok, "30 Hz"), (1000.0, C.aux, "1 kHz"), (5000.0, C.text, "5 kHz")
    ):
        ax.axvline(rate, color=ref, ls=":", lw=1.0, alpha=0.7)
        ax.text(rate * 1.06, len(names) - 1.5, lab, color=ref, fontsize=7.5)
    ax.set_xlim(0.04, 2.0e4)
    ax.set_xlabel("由非 NaN 比例反推的有效采样率 [Hz]（对数轴）")
    ax.set_title("宽表填充率 = 各传感器速率（橙 = 真实产线可得，灰 = 仅仿真）")
    ax.grid(axis="y", alpha=0.2)


def panel_preview(ax, T, seam) -> None:
    t = T["t"]
    gap_true = T["truth_gap"]
    meas, lead = T["prof_gap"], T["prof_lead_s"]
    m = ~np.isnan(meas)
    drop = T["prof_valid"] == 0.0
    ts, gs = downsample(t, gap_true, n=3000)
    ax.plot(ts, gs * 1e3, color=C.truth, lw=1.4, label="真实间隙（在电弧位置）")
    ax.plot(t[m], meas[m] * 1e3, ".", color=C.band_ok, ms=2.6, label="激光轮廓仪（前视）")
    if drop.any():
        ax.plot(
            t[drop], np.full(drop.sum(), -0.18), "|", color=C.bad, ms=7, mew=1.2,
            label=f"飞溅导致丢帧（{int(drop.sum())} 帧，无读数而非错读数）",
        )
        ax.set_ylim(-0.35, None)
    lead_time = float(
        np.nanmedian((lead[m] - T["rb_s"][m]) / np.maximum(T["rb_v_travel"][m], 1e-6))
    )
    ax.set_title(f"前视预览：轮廓仪领先电弧 {lead_time:.1f} s 看到间隙变化")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("root gap [mm]")
    ax.legend(loc="upper left")


def panel_ir_vs_rgb(ax, T) -> None:
    t = T["t"]
    tw, wt = downsample(t, T["truth_pool_w"], n=3000)
    ax.plot(tw, wt * 1e3, color=C.truth, lw=1.6, label="真实熔宽")
    ir = T["ir_pool_width"]
    m = ~np.isnan(ir)
    ax.plot(t[m], ir[m] * 1e3, ".", color=C.warn, ms=3.0, label="IR 热像仪")
    rgb = T["rgb_pool_width"]
    m2 = ~np.isnan(rgb)
    ax.plot(t[m2], rgb[m2] * 1e3, ".", color=C.baseline, ms=3.0, label="RGB 可见光相机")

    def rms(meas):
        k = ~np.isnan(meas) & ~np.isnan(T["truth_pool_w"])
        return float(np.sqrt(np.mean((meas[k] - T["truth_pool_w"][k]) ** 2)) * 1e3)

    ax.set_title(
        f"熔宽测量：IR RMSE {rms(ir):.2f} mm  vs  RGB RMSE {rms(rgb):.2f} mm"
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("pool width [mm]")
    ax.set_ylim(-2, 40)
    ax.legend(loc="upper right", ncols=3)


def panel_rgb_quality(ax, T, cfg) -> None:
    t = T["t"]
    q = T["rgb_quality"]
    m = ~np.isnan(q)
    ax.plot(t[m], q[m], color=C.baseline, lw=1.0, label="RGB 图像可用度")
    ax.axhline(
        cfg.sensors.rgb_quality_min, color=C.bad, ls="--", lw=1.2,
        label=f"可用阈值 {cfg.sensors.rgb_quality_min:g}",
    )
    ax.fill_between(
        t[m], 0, q[m], where=q[m] < cfg.sensors.rgb_quality_min,
        color=C.bad, alpha=0.25, lw=0,
    )
    ax.set_ylabel("quality [-]", color=C.baseline)
    ax.tick_params(axis="y", colors=C.baseline)
    ax2 = ax.twinx()
    ts, sm = downsample(t, T["truth_smoke"], n=3000)
    ax2.plot(ts, sm, color=C.muted, lw=1.0, alpha=0.8)
    ax2.set_ylabel("smoke density [-]", color=C.muted)
    ax2.grid(False)
    ax2.tick_params(colors=C.muted)
    frac = 100.0 * np.mean(q[m] < cfg.sensors.rgb_quality_min)
    ax.set_title(f"RGB 被烟尘与弧光击穿：{frac:.0f} % 的帧不可用")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right")


def panel_acoustics(ax, T) -> None:
    t = T["t"]
    t0, t1 = 4.0, 4.6
    w = (t >= t0) & (t < t1)
    ax.plot((t[w] - t0) * 1e3, T["mic_p"][w], color=C.aux2, lw=1.0, label="mic RMS (per tick)")
    short = T["ps_short"][w] > 0.5
    click = T["mic_click"][w] > 0.5
    top = np.nanmax(T["mic_p"][w])
    ax.fill_between(
        (t[w] - t0) * 1e3, 0, top * 1.15, where=short,
        color=C.bad, alpha=0.25, lw=0, label="短路（电信号）",
    )
    ax.plot(
        (t[w][click] - t0) * 1e3, np.full(click.sum(), top * 1.08), "v",
        color=C.warn, ms=6, label="再引弧声学咔哒",
    )
    ax.set_title("声学与电信号互证：每次短路结束都对应一次声学冲击")
    ax.set_xlabel("time [ms] (600 ms window)")
    ax.set_ylabel("acoustic pressure RMS [Pa]")
    ax.legend(loc="upper right", ncols=3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/phase2_sensors.png"))
    args = ap.parse_args()

    cfg = default_config(seam__kind="step", seam__length=0.20, sim__seed=args.seed)
    use_style()
    res = simulate(cfg, seed=args.seed)
    T = res.table

    fig, axes = plt.subplots(3, 2, figsize=(15.5, 12.5))
    panel_raster(axes[0, 0], T)
    panel_fill(axes[0, 1], T)
    panel_preview(axes[1, 0], T, res.seam)
    panel_ir_vs_rgb(axes[1, 1], T)
    panel_rgb_quality(axes[2, 0], T, cfg)
    panel_acoustics(axes[2, 1], T)

    fig.suptitle(
        "weldloop Phase 2 — 合成传感器套件与统一主时钟日志   "
        f"(seed={args.seed}, {T.n_rows} rows @ {T.f_master:.0f} Hz, "
        f"{len(T.columns)} columns)",
        color=C.text, fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    print(T.summary().splitlines()[0])


if __name__ == "__main__":
    main()
