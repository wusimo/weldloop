"""Figures for the demo: one slide, and one detailed side-by-side dashboard.

``summary_figure`` is deliberately spare — it is meant to be projected, read in
ten seconds, and argued from.  ``dashboard_figure`` is the engineering view:
everything both controllers did, side by side.

Colour rule for the whole repository: charcoal panels, and **orange is
reserved for the adaptive controller**.  Nothing else in any figure is orange.
"""

from __future__ import annotations

import numpy as np

from weldloop.config import WeldConfig
from weldloop.viz.style import C, band, downsample, shade_flag, use_style

__all__ = ["summary_figure", "dashboard_figure"]


def _bin_stat(x: np.ndarray, y: np.ndarray, n: int, stat: str = "mean"):
    """Bin ``y`` against ``x`` into ``n`` bins.

    Plotting 200 000 points of a 5 kHz stream produces a solid smear; binning
    keeps the shape and the file size honest.  ``max``/``min`` preserve the
    excursions that a mean would hide.
    """
    edges = np.linspace(x.min(), x.max(), n + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n - 1)
    out = np.full(n, np.nan)
    order = np.argsort(idx, kind="stable")
    idx_s, y_s = idx[order], y[order]
    bounds = np.searchsorted(idx_s, np.arange(n + 1))
    for i in range(n):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            continue
        seg = y_s[a:b]
        seg = seg[~np.isnan(seg)]
        if seg.size == 0:
            continue
        out[i] = {
            "mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum
        }[stat](seg)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, out


def _history(ctl) -> dict[str, np.ndarray]:
    return {k: np.array([r[k] for r in ctl.history]) for k in ctl.history[0]}


# --------------------------------------------------------------------------
def summary_figure(runs: dict, cfg: WeldConfig, seed: int, gap_profile: str):
    """The slide.

    Top: the disturbance.  Middle: what each controller did about it, with the
    adaptive controller's own belief drawn as a +/-2 sigma band so the audience
    can see that it knew what it was doing.  Bottom: the sensor that made it
    possible — raw 5 kHz V/I with the short-circuit events marked.
    """
    import matplotlib.pyplot as plt

    use_style()
    fig, axes = plt.subplots(
        3, 1, figsize=(14.0, 9.2), height_ratios=[1.0, 2.0, 1.35], sharex=False
    )
    base, adap = runs["baseline"], runs["adaptive"]
    bt, at = base["table"], adap["table"]

    # -- top: the disturbance -------------------------------------------
    ax = axes[0]
    s, gap = downsample(bt["rb_s"] * 1e3, bt["truth_gap"] * 1e3, n=2500)
    ax.fill_between(s, 0, gap, color=C.aux2, alpha=0.28, lw=0)
    ax.plot(s, gap, color=C.aux2, lw=1.6)
    ax.set_ylabel("坡口间隙 [mm]")
    ax.set_title(
        "装配间隙沿焊缝变化 0–4 mm —— 定参数焊接无解的那个扰动",
        loc="left", fontsize=11,
    )
    ax.set_xlim(s.min(), s.max())
    ax.tick_params(labelbottom=False)

    # -- middle: penetration --------------------------------------------
    ax = axes[1]
    h = _history(adap["controller"])
    ax.fill_between(
        h["s"] * 1e3, (h["p_hat"] - 2 * h["p_std"]) * 1e3,
        (h["p_hat"] + 2 * h["p_std"]) * 1e3,
        color=C.adaptive, alpha=0.17, lw=0, label="自适应控制器的估计 ±2σ",
    )
    band(ax, cfg.control.p_lo * 1e3, cfg.control.p_hi * 1e3, label="验收带 3–5 mm")
    sb, pb, btf = downsample(
        bt["rb_s"] * 1e3, bt["truth_penetration"] * 1e3, bt["truth_burn_through"], n=2500
    )
    ax.plot(sb, pb, color=C.baseline, lw=1.6, label="定参数：真实熔深")
    hit = btf > 0.5
    if hit.any():
        ax.plot(sb[hit], pb[hit], "o", color=C.bad, ms=4.5, mew=0, label="烧穿")
    sa, pa = downsample(at["rb_s"] * 1e3, at["truth_penetration"] * 1e3, n=2500)
    ax.plot(sa, pa, color=C.adaptive, lw=1.8, label="自适应：真实熔深")
    ax.axhline(
        cfg.joint.thickness * 1e3, color=C.bad, ls="--", lw=1.3, label="板厚 6 mm"
    )
    mb, ma = base["metrics"], adap["metrics"]
    ax.set_ylabel("熔深 [mm]")
    ax.set_ylim(2.2, 6.5)
    ax.set_xlim(sb.min(), sb.max())
    ax.set_title(
        f"定参数 {mb.burn_through_events} 处烧穿 / {mb.burn_through_length_mm:.1f} mm，"
        f"熔深 std {mb.penetration_std_mm:.2f} mm，带内 {mb.penetration_in_band_pct:.0f} %"
        f"      →      "
        f"自适应 {ma.burn_through_events} 处 / {ma.burn_through_length_mm:.1f} mm，"
        f"std {ma.penetration_std_mm:.2f} mm，带内 {ma.penetration_in_band_pct:.0f} %"
        f"（循环时间 {mb.duration_s:.0f} → {ma.duration_s:.0f} s）",
        loc="left", fontsize=10.5,
    )
    ax.plot(
        h["s"] * 1e3, h["p_hat"] * 1e3, color=C.adaptive, lw=0.9, alpha=0.75,
        label="自适应：在线估计均值",
    )
    ax.legend(loc="lower left", ncols=3, fontsize=8.5)
    ax.tick_params(labelbottom=False)

    # -- bottom: the sensor ---------------------------------------------
    # Deliberately the BASELINE run: with fixed parameters the gap swings by
    # 4 mm and the machine's own meters barely move.  That is the problem
    # statement for Phase 3, drawn.
    ax = axes[2]
    s_b = bt["rb_s"] * 1e3
    arcing = bt["ps_short"] < 0.5
    cs, V = _bin_stat(s_b[arcing], bt["ps_V"][arcing], 600)
    _, I = _bin_stat(s_b[arcing], bt["ps_I"][arcing], 600)
    ax.plot(cs, V, color=C.aux2, lw=1.3, label="电弧电压 V")
    ax.set_ylabel("V [V]", color=C.aux2)
    ax.tick_params(axis="y", colors=C.aux2)
    ax.set_ylim(np.nanmean(V) - 3.0, np.nanmean(V) + 3.0)
    ax2 = ax.twinx()
    ax2.plot(cs, I, color=C.warn, lw=1.3, label="电流 I")
    ax2.set_ylabel("I [A]", color=C.warn)
    ax2.set_ylim(np.nanmean(I) - 40.0, np.nanmean(I) + 40.0)
    ax2.tick_params(colors=C.warn)
    ax2.grid(False)
    ax.set_xlim(cs.min(), cs.max())
    ax.set_xlabel("沿焊缝位置 s [mm]")
    ax.set_title(
        f"定参数焊接下的主传感器读数：间隙从 0 走到 4 mm，"
        f"电压 {np.nanmean(V):.1f} ± {np.nanstd(V):.2f} V、电流 {np.nanmean(I):.0f} ± {np.nanstd(I):.1f} A —— "
        f"仪表上几乎什么也没发生。熔深信息在波形结构里（Phase 3）。",
        loc="left", fontsize=10,
    )

    # inset: 120 ms of the raw stream, where the information actually is
    ins = ax.inset_axes((0.685, 0.44, 0.30, 0.50))
    dt = 1.0 / bt.f_master
    n0 = bt.n_rows // 2
    n = int(0.12 / dt)
    tt = np.arange(n) * dt * 1e3
    Vw = bt["ps_V"][n0 : n0 + n]
    shorted = bt["ps_short"][n0 : n0 + n] > 0.5
    ins.plot(tt, Vw, color=C.aux2, lw=0.7)
    if shorted.any():
        ins.plot(tt[shorted], Vw[shorted], ".", color=C.bad, ms=1.8)
    ins.set_facecolor(C.bg)
    ins.set_title("原始 5 kHz，120 ms", color=C.text, fontsize=7.5, pad=2)
    ins.tick_params(labelsize=6, colors=C.muted)
    ins.set_xlabel("ms", fontsize=6.5, labelpad=1)
    for spine in ins.spines.values():
        spine.set_color(C.grid)

    fig.suptitle(
        "电源在环自适应焊接 —— 变间隙厚板 MIG/MAG    "
        f"seed={seed}, {gap_profile} 间隙, 6 mm 低碳钢对接（全部数字由本仓库代码运行产生）",
        color=C.text, fontsize=13.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    return fig


# --------------------------------------------------------------------------
def dashboard_figure(runs: dict, cfg: WeldConfig):
    """The engineering view: both controllers, four rows, same axes."""
    import matplotlib.pyplot as plt

    use_style()
    names = [n for n in ("baseline", "adaptive") if n in runs]
    fig, axes = plt.subplots(4, len(names), figsize=(15.5, 13.0), squeeze=False)

    for col, name in enumerate(names):
        d = runs[name]
        t = d["table"]
        colour = C.adaptive if name == "adaptive" else C.baseline
        s = t["rb_s"] * 1e3

        # row 0: gap and travel speed
        ax = axes[0][col]
        x, gap = downsample(s, t["truth_gap"] * 1e3, n=2000)
        ax.plot(x, gap, color=C.aux2, lw=1.3, label="坡口间隙")
        ax.set_ylabel("gap [mm]", color=C.aux2)
        ax.tick_params(axis="y", colors=C.aux2)
        ax2 = ax.twinx()
        _, v = downsample(s, t["rb_v_travel"] * 1e3, n=2000)
        ax2.plot(x, v, color=colour, lw=1.3, label="行走速度")
        ax2.set_ylabel("v [mm/s]")
        ax2.set_ylim(1.5, 14.5)
        ax2.grid(False)
        ax.set_title(f"{name}：扰动与行走速度")
        ax.legend(loc="upper left", fontsize=7.5)
        ax2.legend(loc="upper right", fontsize=7.5)

        # row 1: penetration with defect flags
        ax = axes[1][col]
        x, p, btf, loff = downsample(
            s, t["truth_penetration"] * 1e3, t["truth_burn_through"],
            t["truth_lack_of_fusion"], n=2500,
        )
        shade_flag(ax, x, btf > 0.5, color=C.bad, label="烧穿")
        shade_flag(ax, x, loff > 0.5, color=C.warn, alpha=0.22, label="未熔合")
        band(ax, cfg.control.p_lo * 1e3, cfg.control.p_hi * 1e3)
        ax.plot(x, p, color=colour, lw=1.4, label="真实熔深")
        ax.axhline(cfg.joint.thickness * 1e3, color=C.bad, ls="--", lw=1.1)
        ax.set_ylim(2.0, 6.6)
        ax.set_ylabel("penetration [mm]")
        m = d["metrics"]
        ax.set_title(f"缺陷：烧穿 {m.burn_through_length_mm:.1f} mm，未熔合 {m.lack_of_fusion_length_mm:.1f} mm")
        ax.legend(loc="lower right", ncols=3, fontsize=7.5)

        # row 2: commands
        ax = axes[2][col]
        x, I, weave = downsample(s, t["cmd_I_set"], t["cmd_weave_amp"] * 1e3, n=2000)
        ax.plot(x, I, color=colour, lw=1.3, label="电流指令")
        ax.set_ylabel("I [A]", color=colour)
        ax.tick_params(axis="y", colors=colour)
        ax.set_ylim(160, 300)
        ax2 = ax.twinx()
        ax2.plot(x, weave, color=C.aux, lw=1.3, label="摆幅指令")
        ax2.set_ylabel("weave [mm]")
        ax2.set_ylim(-0.2, 4.4)
        ax2.grid(False)
        ax.set_title("控制指令")
        ax.legend(loc="upper left", fontsize=7.5)
        ax2.legend(loc="upper right", fontsize=7.5)

        # row 3: estimate vs truth (adaptive only has an estimator)
        ax = axes[3][col]
        x, p = downsample(s, t["truth_penetration"] * 1e3, n=2500)
        ax.plot(x, p, color=C.truth, lw=1.2, ls="--", label="真值")
        ctl = d["controller"]
        if getattr(ctl, "history", None):
            h = _history(ctl)
            ax.fill_between(
                h["s"] * 1e3, (h["p_hat"] - 2 * h["p_std"]) * 1e3,
                (h["p_hat"] + 2 * h["p_std"]) * 1e3,
                color=colour, alpha=0.22, lw=0, label="估计 ±2σ",
            )
            ax.plot(h["s"] * 1e3, h["p_hat"] * 1e3, color=colour, lw=1.3, label="估计均值")
            rmse = float(
                np.sqrt(np.mean((np.interp(h["s"], t["rb_s"], t["truth_penetration"]) - h["p_hat"]) ** 2))
            )
            ax.set_title(f"在线估计（RMSE {rmse * 1e3:.2f} mm）")
        else:
            ax.text(
                0.5, 0.5, "定参数控制器没有估计器：\n它对熔池一无所知",
                transform=ax.transAxes, ha="center", va="center",
                color=C.muted, fontsize=11,
            )
            ax.set_title("在线估计")
        band(ax, cfg.control.p_lo * 1e3, cfg.control.p_hi * 1e3)
        ax.set_ylim(2.0, 6.6)
        ax.set_xlabel("沿焊缝位置 s [mm]")
        ax.set_ylabel("penetration [mm]")
        ax.legend(loc="lower right", ncols=3, fontsize=7.5)

    fig.suptitle(
        "weldloop 控制对比仪表板 —— 定参数 vs 自适应",
        color=C.text, fontsize=13, fontweight="bold", y=0.996,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.988))
    return fig
