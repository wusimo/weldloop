"""The planning-layer figure: where the seam got cut, and what it cost.

Three panels and a ledger, in the demo's usual palette:

1. the joint — measured gap, plate thickness, and where each planner decided
   one set of parameters stops being right;
2. penetration against an acceptance band that *steps with the plate*, which
   is the whole reason the thickness note matters;
3. what the cell actually did — commanded current and travel speed;
4. burn-through length per arm, plus the latency ledger that says why this
   layer is offline.
"""

from __future__ import annotations

import numpy as np

from weldloop.viz.style import C, band, downsample, shade_flag, use_style

__all__ = ["vla_figure"]

_ARM_COLORS = {
    "baseline": C.baseline,
    "adaptive": "#9aa6b2",
    "rule+adapt": C.aux,
    "vla-plan": C.aux2,
    "vla+adapt": C.adaptive,
    "vla-noH+adapt": C.warn,
}


def vla_figure(runs: dict, seam, cfg, plan, trace, rule_plan=None,
               n_decisions: int = 0):
    """Build the planning figure.

    ``runs`` maps arm name -> ``{"table":..., "metrics":...}``.
    """
    import matplotlib.pyplot as plt

    use_style()
    fig = plt.figure(figsize=(16.0, 9.5))
    gs = fig.add_gridspec(
        4, 1, height_ratios=[1.15, 1.35, 1.0, 1.15],
        left=0.062, right=0.985, top=0.925, bottom=0.062, hspace=0.42,
    )

    s_mm = seam.s * 1e3
    h_mm = seam.thickness * 1e3
    L = float(s_mm[-1])
    frac_lo = cfg.control.p_lo / cfg.joint.thickness
    frac_hi = cfg.control.p_hi / cfg.joint.thickness

    fig.suptitle(
        "R4 任务规划层：视觉-语言模型读装配报告和工艺卡，切分焊缝、给出发点   "
        "task planning by a VLA, offline",
        fontsize=14, fontweight="bold", y=0.975,
    )

    # -- 1. the joint and where it was cut --------------------------------
    ax = fig.add_subplot(gs[0])
    ax.plot(s_mm, seam.gap * 1e3, color=C.truth, lw=1.5, label="实测根部间隙 measured gap")
    ax.set_ylabel("间隙 gap [mm]")
    ax.set_xlim(0, L)
    ax.set_ylim(0, max(5.0, float(seam.gap.max() * 1e3) * 1.55))

    ax2 = ax.twinx()
    ax2.step(s_mm, h_mm, where="post", color=_ARM_COLORS["vla-plan"], lw=1.6,
             label="板厚 plate thickness（图纸，扫描看不到）")
    ax2.set_ylabel("板厚 h [mm]", color=_ARM_COLORS["vla-plan"])
    ax2.set_ylim(0, float(h_mm.max()) * 1.9)
    ax2.grid(False)
    ax2.tick_params(colors=_ARM_COLORS["vla-plan"])

    for edge in [s.s_start * 1e3 for s in plan.segments[1:]]:
        ax.axvline(edge, color=C.adaptive, lw=1.6, ls="-", alpha=0.9)
    if rule_plan is not None:
        for edge in [s.s_start * 1e3 for s in rule_plan.segments[1:]]:
            ax.axvline(edge, color=C.aux, lw=1.4, ls=(0, (5, 3)), alpha=0.9)
    for seg in plan.segments:
        ax.text(0.5 * (seg.s_start + seg.s_end) * 1e3, ax.get_ylim()[1] * 0.97,
                f"{seg.gap_class}\n{seg.I_set:.0f} A / {seg.v_travel * 1e3:.1f} mm/s\n"
                f"h = {seg.thickness * 1e3 if seg.thickness else 0:.0f} mm",
                color=C.adaptive, fontsize=8.0, ha="center", va="top", linespacing=1.35)

    handles = [
        plt.Line2D([], [], color=C.truth, lw=1.5, label="实测根部间隙 measured gap"),
        plt.Line2D([], [], color=_ARM_COLORS["vla-plan"], lw=1.6, label="板厚 thickness"),
        plt.Line2D([], [], color=C.adaptive, lw=1.6, label="VLA 分段 VLA cuts"),
    ]
    if rule_plan is not None:
        handles.append(plt.Line2D([], [], color=C.aux, lw=1.4, ls=(0, (5, 3)),
                                  label="规则分段（等分）rule cuts"))
    ax.legend(handles=handles, loc="lower right", ncol=2)
    ax.set_title("① 接头与分段：规则规划器等分，VLA 切在工况真正变化的地方",
                 loc="left")

    # -- 2. penetration against a band that steps -------------------------
    ax = fig.add_subplot(gs[1])
    ax.fill_between(s_mm, frac_lo * h_mm, frac_hi * h_mm,
                    color=C.band_ok, alpha=0.14, lw=0,
                    label="验收带 = 板厚的固定比例 acceptance band")
    ax.plot(s_mm, h_mm, color=C.bad, lw=1.0, ls=(0, (2, 3)), alpha=0.8,
            label="板厚 plate thickness")
    for name, run in runs.items():
        t = run["table"]
        x, p = downsample(t["rb_s"] * 1e3, t["truth_penetration"] * 1e3)
        ax.plot(x, p, color=_ARM_COLORS.get(name, C.muted), lw=1.5,
                alpha=0.95 if name == "vla+adapt" else 0.75, label=name)
    ax.set_xlim(0, L)
    ax.set_ylim(0, float(h_mm.max()) * 1.25)
    ax.set_ylabel("熔深 penetration [mm]")
    ax.legend(loc="lower left", ncol=4)
    ax.set_title("② 熔深：验收带随板厚下降，只有读到板厚的那一路跟着降下来", loc="left")

    # -- 3. what the cell did ---------------------------------------------
    ax = fig.add_subplot(gs[2])
    head = runs.get("vla+adapt") or next(iter(runs.values()))
    t = head["table"]
    x, I = downsample(t["rb_s"] * 1e3, t["cmd_I_set"])
    ax.plot(x, I, color=C.adaptive, lw=1.4, label="电流指令 I_cmd [A]")
    ax.set_ylabel("电流 I [A]", color=C.adaptive)
    ax.set_xlim(0, L)
    axv = ax.twinx()
    x, v = downsample(t["rb_s"] * 1e3, t["rb_v_travel"] * 1e3)
    axv.plot(x, v, color=C.aux2, lw=1.4, label="焊接速度 v [mm/s]")
    axv.set_ylabel("速度 v [mm/s]", color=C.aux2)
    axv.grid(False)
    axv.tick_params(colors=C.aux2)
    shade_flag(ax, t["rb_s"] * 1e3, t["truth_burn_through"] > 0.5, color=C.bad)
    for edge in [s.s_start * 1e3 for s in plan.segments[1:]]:
        ax.axvline(edge, color=C.muted, lw=1.0, alpha=0.6)
    ax.set_title("③ vla+adapt 实际执行：计划是出发点，50 Hz 层继续调（红=烧穿）",
                 loc="left")

    # -- 4. scoreboard + latency ledger -----------------------------------
    ax = fig.add_subplot(gs[3])
    names = list(runs.keys())
    bt = [runs[n]["metrics"].burn_through_length_mm for n in names]
    colors = [_ARM_COLORS.get(n, C.muted) for n in names]
    ypos = np.arange(len(names))
    ax.barh(ypos, bt, color=colors, height=0.62)
    for y, value in zip(ypos, bt):
        ax.text(value + max(bt) * 0.015, y, f"{value:.1f} mm", va="center",
                fontsize=9, color=C.text)
    ax.set_yticks(ypos, names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("烧穿长度 burn-through length [mm]（越短越好）")
    top = max(bt) if max(bt) > 0 else 1.0
    ax.set_xlim(0, top * 2.05)
    ax.set_xticks(np.linspace(0.0, np.ceil(top / 25.0) * 25.0,
                              int(np.ceil(top / 25.0)) + 1))
    ax.set_title("④ 记分板", loc="left")

    live = not trace.backend.startswith("recorded")
    stamp = f"{trace.latency_s:.2f} s" if live else f"{trace.latency_s:.2f} s (回放 replay)"
    ledger = (
        "规划层 planning layer（离线 offline）\n"
        f"  调用 calls              1，起弧前 before the arc\n"
        f"  输出 outputs            {trace.n_segments} 段 segments\n"
        f"  后端 backend            {trace.backend} / {trace.model}\n"
        f"  耗时 latency            {stamp}\n"
        f"  被夹的设定点 clamped      {trace.n_clamped}\n"
        "\n运动层 motion layer（实时 real time）\n"
        f"  周期 period             {cfg.control.dt * 1e3:.0f} ms\n"
        f"  本次决策 decisions       {n_decisions}\n"
        f"  每段决策 per segment     {n_decisions / max(trace.n_segments, 1):.0f}"
    )
    import matplotlib as mpl

    ax.text(0.60, 1.02, ledger, transform=ax.transAxes, fontsize=9.0,
            family=list(mpl.rcParams["font.monospace"]), color=C.muted,
            va="top", ha="left", linespacing=1.45)
    return fig
