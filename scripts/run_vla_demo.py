#!/usr/bin/env python3
"""The planning demo: what a vision-language model buys at the task layer.

    python scripts/run_vla_demo.py

One stepped-plate seam (6 mm for 120 mm, then 4 mm), welded six ways.  The
thickness step is on the drawing and in the job card; it is *not* in the laser
scan, so it reaches the cell only if something reads prose.

    baseline        one parameter set for the whole seam — today's cell
    adaptive        50 Hz closed loop, no plan
    rule+adapt      rule planner with the full measured scan + closed loop
    vla-plan        the VLA's plan, executed open-loop
    vla+adapt       the VLA's plan as feed-forward + closed loop
    vla-noH+adapt   ablation: the same plan with the thickness note removed

The last arm is the control that matters.  It keeps the VLA's segmentation and
every setpoint and changes one thing — each segment is declared at the job's
nominal 6 mm — so the difference between it and ``vla+adapt`` is the value of
one sentence of prose, not of the model's arithmetic.

Writes ``out/vla_plan.png`` and ``out/vla_metrics.json``.  Takes ~2 min.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from weldloop.control.baseline import BaselineController
from weldloop.control.motion_layer import AdaptiveController
from weldloop.control.planned import PlannedController
from weldloop.metrics import score
from weldloop.planning.job import (
    DEMO_JOB, FitupScan, demo_config, demo_seam, demo_seam_description,
)
from weldloop.planning.task_planner import TaskPlanner
from weldloop.planning.vla import VLATaskPlanner, resolve_backend
from weldloop.sim.runner import simulate
from weldloop.viz.vla import vla_figure


def strip_thickness(plan, cfg):
    """The ablation: same plan, thickness note never read."""
    return replace(plan, segments=tuple(
        replace(seg, thickness=cfg.joint.thickness, p_target=cfg.control.p_target,
                p_lo=cfg.control.p_lo, p_hi=cfg.control.p_hi)
        for seg in plan.segments
    ))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--live", action="store_true", help="call the model, don't replay")
    ap.add_argument("--scan", type=Path, default=Path("docs/vla_fitup_scan.png"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    t_start = time.time()
    cfg = demo_config(seed=args.seed)
    seam = demo_seam(cfg, seed=args.seed)
    desc = demo_seam_description(cfg)

    if not args.scan.is_file():
        print(f"! {args.scan} is missing — run scripts/render_fitup_scan.py first")
        return 1

    rule_plan = TaskPlanner(cfg).plan(desc, measured=seam)
    planner = VLATaskPlanner(cfg, backend=resolve_backend(prefer_live=args.live))
    vla_plan, trace = planner.plan(DEMO_JOB, FitupScan(args.scan), desc, measured=seam)
    if trace.fallback:
        print(f"! planner fell back to rules: {trace.fallback}")

    print(f"seam: 6 mm -> {cfg.seam.thickness_thin * 1e3:.0f} mm stepped plate, "
          f"{cfg.seam.length * 1e3:.0f} mm, gap 0 - {cfg.seam.gap_max * 1e3:.0f} mm, "
          f"seed {args.seed}")
    print(f"planner: {trace.backend} / {trace.model}, {trace.n_segments} segments, "
          f"{trace.n_clamped} setpoint(s) clamped\n")

    arms = {
        "baseline":      BaselineController(cfg),
        "adaptive":      AdaptiveController(cfg, name="adaptive"),
        "rule+adapt":    AdaptiveController(cfg, plan=rule_plan, name="rule+adapt"),
        "vla-plan":      PlannedController(cfg, vla_plan, name="vla-plan"),
        "vla+adapt":     AdaptiveController(cfg, plan=vla_plan, name="vla+adapt"),
        "vla-noH+adapt": AdaptiveController(cfg, plan=strip_thickness(vla_plan, cfg),
                                            name="vla-noH+adapt"),
    }

    runs, n_decisions = {}, 0
    for name, ctl in arms.items():
        t0 = time.time()
        res = simulate(cfg, seed=args.seed, controller=ctl, seam=seam)
        runs[name] = {"table": res.table, "metrics": score(res.table, cfg, name)}
        if name == "vla+adapt":
            n_decisions = len(getattr(ctl, "history", []))
        print(f"  {runs[name]['metrics'].headline()}   [{time.time() - t0:.0f} s]")

    args.out.mkdir(parents=True, exist_ok=True)
    fig = vla_figure(runs, seam, cfg, vla_plan, trace, rule_plan=rule_plan,
                     n_decisions=n_decisions)
    fig.savefig(args.out / "vla_plan.png", dpi=130)
    plt.close(fig)

    base = runs["baseline"]["metrics"]
    best = runs["vla+adapt"]["metrics"]
    abl = runs["vla-noH+adapt"]["metrics"]
    payload = {
        "seed": args.seed,
        "job": DEMO_JOB.job_id,
        "plate": {"thick_mm": cfg.joint.thickness * 1e3,
                  "thin_mm": cfg.seam.thickness_thin * 1e3,
                  "step_at_mm": cfg.seam.thickness_step_at * cfg.seam.length * 1e3},
        "planner": trace.as_dict(),
        "plan": vla_plan.as_dict(),
        "motion_layer_decisions": n_decisions,
        "control_dt_ms": cfg.control.dt * 1e3,
        "arms": {n: d["metrics"].as_dict() for n, d in runs.items()},
        "headline": {
            "burn_through_mm_baseline": base.burn_through_length_mm,
            "burn_through_mm_vla_adapt": best.burn_through_length_mm,
            "burn_through_mm_ablation": abl.burn_through_length_mm,
            "in_band_pct_baseline": base.penetration_in_band_pct,
            "in_band_pct_vla_adapt": best.penetration_in_band_pct,
            "cycle_time_s_baseline": base.duration_s,
            "cycle_time_s_vla_adapt": best.duration_s,
        },
        "wall_clock_s": None,
    }
    payload["wall_clock_s"] = round(time.time() - t_start, 1)
    (args.out / "vla_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"  burn-through  baseline {base.burn_through_length_mm:6.1f} mm"
          f"  ->  vla+adapt {best.burn_through_length_mm:5.1f} mm")
    print(f"                ablation (same plan, thickness note removed): "
          f"{abl.burn_through_length_mm:5.1f} mm")
    print(f"  in band       {base.penetration_in_band_pct:6.1f} %"
          f"  ->  {best.penetration_in_band_pct:5.1f} %")
    print(f"  cycle time    {base.duration_s:6.1f} s   ->  {best.duration_s:5.1f} s")
    print(f"  planner produced {trace.n_segments} segments; the motion layer made "
          f"{n_decisions} decisions in the same weld")
    print()
    for f in ("vla_plan.png", "vla_metrics.json"):
        print(f"  wrote {args.out / f}")
    print(f"\ntotal {payload['wall_clock_s']:.0f} s wall clock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
