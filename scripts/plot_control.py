#!/usr/bin/env python3
"""Phase 4 figure: fixed parameters vs adaptive vs RGB vision.

All three arms weld the same seam with the same physics and the same noise
realisation.  The x-axis is position along the seam, not time, because the
three arms take different amounts of time to get there — which is itself one
of the results.

Panels:

1. The disturbance and what each controller did about the travel speed.
2. True penetration for all three arms, with the acceptance band and the
   burn-through markers.  This is the slide.
3. What the adaptive layer moved: current and weave against the gap.
4. The metrics table as bars.
5. The adaptive arm's own view: estimate +/- 2 sigma against truth.
6. The vision arm's view — the same picture, with a filter that cannot see.

Usage::

    python scripts/plot_control.py --seed 0 --out out/phase4_control.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from weldloop.config import default_config
from weldloop.control.baseline import BaselineController
from weldloop.control.motion_layer import AdaptiveController
from weldloop.control.vision import VisionController
from weldloop.estimation.residual import load_residual
from weldloop.metrics import score
from weldloop.sim.runner import simulate
from weldloop.viz.style import C, band, downsample, shade_flag, use_style

ARM_COLOR = {
    "baseline": C.baseline,
    "adaptive": C.adaptive,
    "vision": C.bad,
    "adaptive+residual": C.band_ok,
}
ARM_LABEL = {
    "baseline": "定参数 baseline",
    "adaptive": "自适应 adaptive",
    "vision": "RGB 视觉 vision",
    "adaptive+residual": "自适应 + 学习残差",
}


def run_arms(cfg, seed: int, residual=None) -> dict:
    arms = {
        "baseline": BaselineController(cfg),
        "vision": VisionController(cfg),
        "adaptive": AdaptiveController(cfg),
    }
    if residual is not None:
        arms["adaptive+residual"] = AdaptiveController(
            cfg, residual=residual, name="adaptive+residual"
        )
    out = {}
    for name, ctl in arms.items():
        t0 = time.time()
        res = simulate(cfg, seed=seed, controller=ctl)
        out[name] = {
            "table": res.table,
            "metrics": score(res.table, cfg, name),
            "controller": ctl,
            "wall_s": time.time() - t0,
        }
        print(f"  {out[name]['metrics'].headline()}   [{out[name]['wall_s']:.1f} s]")
    return out


def panel_speed(ax, arms, cfg) -> None:
    base = arms["baseline"]["table"]
    s, gap = downsample(base["rb_s"] * 1e3, base["truth_gap"] * 1e3, n=2500)
    ax.plot(s, gap, color=C.aux2, lw=1.5, label="坡口间隙 g(s)")
    ax.set_ylabel("root gap [mm]", color=C.aux2)
    ax.tick_params(axis="y", colors=C.aux2)
    ax2 = ax.twinx()
    for name, d in arms.items():
        t = d["table"]
        x, v = downsample(t["rb_s"] * 1e3, t["rb_v_travel"] * 1e3, n=2500)
        ax2.plot(x, v, color=ARM_COLOR[name], lw=1.3, label=ARM_LABEL[name])
    ax2.set_ylabel("travel speed [mm/s]")
    ax2.grid(False)
    ax.set_xlabel("position along seam s [mm]")
    ax.set_title("扰动与各控制器的行走速度响应")
    ax2.legend(loc="upper right", ncols=2, fontsize=7.5)
    ax.legend(loc="upper left", fontsize=7.5)


def panel_penetration(ax, arms, cfg) -> None:
    for name, d in arms.items():
        t = d["table"]
        x, p, bt = downsample(
            t["rb_s"] * 1e3, t["truth_penetration"] * 1e3,
            t["truth_burn_through"], n=3000,
        )
        ax.plot(x, p, color=ARM_COLOR[name], lw=1.4, label=ARM_LABEL[name])
        hit = bt > 0.5
        if hit.any():
            ax.plot(x[hit], p[hit], "o", color=ARM_COLOR[name], ms=3.5, mew=0)
    band(ax, cfg.control.p_lo * 1e3, cfg.control.p_hi * 1e3, label="验收带")
    ax.axhline(
        cfg.joint.thickness * 1e3, color=C.bad, ls="--", lw=1.2, label="板厚 = 烧穿"
    )
    ax.set_xlabel("position along seam s [mm]")
    ax.set_ylabel("true penetration [mm]")
    ax.set_ylim(2.0, 6.6)
    ax.set_title("真实熔深（圆点 = 烧穿判据触发）")
    ax.legend(loc="lower right", ncols=2, fontsize=7.5)


def panel_handles(ax, arms, cfg) -> None:
    d = arms["adaptive"]["table"]
    x, I, weave, gap = downsample(
        d["rb_s"] * 1e3, d["cmd_I_set"], d["cmd_weave_amp"] * 1e3,
        d["truth_gap"] * 1e3, n=2500,
    )
    ax.plot(x, I, color=C.adaptive, lw=1.3, label="电流指令 I_cmd")
    b = arms["baseline"]["table"]
    ax.axhline(cfg.baseline.I_set, color=C.baseline, ls=":", lw=1.2, label="baseline 固定电流")
    ax.set_ylabel("current [A]", color=C.adaptive)
    ax.tick_params(axis="y", colors=C.adaptive)
    ax2 = ax.twinx()
    ax2.plot(x, weave, color=C.aux, lw=1.3, label="摆幅指令")
    ax2.plot(x, gap, color=C.aux2, lw=1.0, ls="--", alpha=0.8, label="坡口间隙")
    ax2.set_ylabel("weave half-amplitude / gap [mm]")
    ax2.grid(False)
    ax.set_xlabel("position along seam s [mm]")
    ax.set_title("自适应层动了哪些手柄：间隙张开 → 降流 + 加摆 + 减速")
    ax.legend(loc="upper left", fontsize=7.5)
    ax2.legend(loc="upper right", fontsize=7.5)


def panel_metrics(ax, arms) -> None:
    names = list(arms)
    fields = [
        ("burn_through_length_mm", "烧穿长度 [mm]"),
        ("lack_of_fusion_length_mm", "未熔合长度 [mm]"),
        ("penetration_std_mm", "熔深标准差 [mm]"),
        ("duration_s", "循环时间 [s]"),
    ]
    floor = 0.02
    x = np.arange(len(fields))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        m = arms[name]["metrics"].as_dict()
        vals = [max(m[f], floor) for f, _ in fields]
        pos = x + (i - (len(names) - 1) / 2) * w
        ax.bar(pos, vals, width=w * 0.92, color=ARM_COLOR[name], label=ARM_LABEL[name],
               log=True)
        for xi, v, (f, _) in zip(pos, vals, fields):
            ax.text(xi, v * 1.15, f"{m[f]:.1f}", ha="center", color=C.text, fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in fields], fontsize=8)
    ax.set_ylim(floor * 0.6, 900.0)
    ax.set_ylabel("对数刻度（跨度太大）")
    ax.set_title("指标对比（越低越好）")
    ax.legend(loc="upper left", fontsize=7.5, ncols=2)


def panel_belief(ax, arms, cfg, name: str) -> None:
    d = arms[name]
    ctl = d["controller"]
    t = d["table"]
    h = {k: np.array([r[k] for r in ctl.history]) for k in ctl.history[0]}
    s_true, p_true = downsample(t["rb_s"] * 1e3, t["truth_penetration"] * 1e3, n=3000)
    ax.plot(s_true, p_true, color=C.truth, lw=1.3, ls="--", label="真值")
    col = ARM_COLOR[name]
    ax.fill_between(
        h["s"] * 1e3, (h["p_hat"] - 2 * h["p_std"]) * 1e3,
        (h["p_hat"] + 2 * h["p_std"]) * 1e3,
        color=col, alpha=0.22, lw=0, label="控制器相信的 ±2σ",
    )
    ax.plot(h["s"] * 1e3, h["p_hat"] * 1e3, color=col, lw=1.3, label="控制器相信的均值")
    if h["emergency"].any():
        shade_flag(ax, h["s"] * 1e3, h["emergency"] > 0.5, color=C.warn, alpha=0.25,
                   label="烧穿预警触发")
    band(ax, cfg.control.p_lo * 1e3, cfg.control.p_hi * 1e3)
    ax.axhline(cfg.joint.thickness * 1e3, color=C.muted, ls=":", lw=1.0)
    m = d["metrics"]
    ax.set_ylim(0.0, 7.5)
    ax.set_xlabel("position along seam s [mm]")
    ax.set_ylabel("penetration [mm]")
    ax.set_title(
        f"{ARM_LABEL[name]}：控制器眼中的世界   "
        f"（无缺陷长度 {m.defect_free_length_pct:.0f} %）"
    )
    ax.legend(loc="upper left", ncols=2, fontsize=7.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-profile", type=str, default="step")
    ap.add_argument("--length", type=float, default=0.20)
    ap.add_argument("--residual", type=Path, default=Path("data/residual.pt"))
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("out/phase4_control.png"))
    args = ap.parse_args()

    cfg = default_config(
        seam__kind=args.gap_profile, seam__length=args.length, sim__seed=args.seed
    )
    use_style()
    residual = None if args.no_residual else load_residual(args.residual)
    arms = run_arms(cfg, args.seed, residual)

    fig, axes = plt.subplots(3, 2, figsize=(15.5, 13.5))
    panel_speed(axes[0, 0], arms, cfg)
    panel_penetration(axes[0, 1], arms, cfg)
    panel_handles(axes[1, 0], arms, cfg)
    panel_metrics(axes[1, 1], arms)
    panel_belief(axes[2, 0], arms, cfg, "adaptive")
    panel_belief(axes[2, 1], arms, cfg, "vision")

    fig.suptitle(
        "weldloop Phase 4 — 定参数 vs 自适应 vs RGB 视觉   "
        f"(seed={args.seed}, {args.gap_profile} 间隙 0–4 mm, 6 mm 板)",
        color=C.text, fontsize=13, fontweight="bold", y=0.996,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.988))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    rows = [arms[n]["metrics"].as_dict() for n in arms]
    (args.out.parent / "phase4_metrics.json").write_text(
        json.dumps({"seed": args.seed, "gap_profile": args.gap_profile, "arms": rows}, indent=2)
    )
    print(f"wrote {args.out.parent / 'phase4_metrics.json'}")


if __name__ == "__main__":
    main()
