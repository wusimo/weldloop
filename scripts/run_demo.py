#!/usr/bin/env python3
"""The demo.  One command, no hardware, under a minute on a laptop.

    python scripts/run_demo.py --seed 0 --gap-profile step

Simulates one variable-gap seam, welds it twice — once with fixed parameters,
once with the adaptive power-source-in-the-loop controller — and writes:

    out/summary.png     the slide: gap profile / penetration / raw V/I
    out/dashboard.png   the engineering view, both controllers side by side
    out/metrics.json    every number in the README tables

Add ``--vision`` for the RGB-camera control arm, and ``--residual`` to use a
trained residual network if one exists (the demo runs on pure physics without
it).  Everything is deterministic given ``--seed``.
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

from weldloop.config import default_config
from weldloop.control.baseline import BaselineController
from weldloop.control.motion_layer import AdaptiveController
from weldloop.control.vision import VisionController
from weldloop.estimation.residual import load_residual
from weldloop.metrics import score
from weldloop.sim.runner import simulate
from weldloop.viz.dashboard import dashboard_figure, summary_figure

GAP_PROFILES = ("step", "ramp", "sine", "random", "constant")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-profile", choices=GAP_PROFILES, default="step")
    ap.add_argument("--length", type=float, default=0.20, help="seam length [m]")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--vision", action="store_true",
                    help="also run the RGB-camera control arm")
    ap.add_argument("--residual", type=Path, default=Path("data/residual.pt"))
    ap.add_argument("--no-residual", action="store_true",
                    help="ignore any trained residual and run pure physics")
    args = ap.parse_args()

    t_start = time.time()
    cfg = default_config(
        seam__kind=args.gap_profile, seam__length=args.length, sim__seed=args.seed
    )
    residual = None if args.no_residual else load_residual(args.residual)

    controllers = {
        "baseline": BaselineController(cfg),
        "adaptive": AdaptiveController(cfg, residual=residual),
    }
    if args.vision:
        controllers["vision"] = VisionController(cfg)

    print(
        f"seam: {args.gap_profile}, {args.length * 1e3:.0f} mm, "
        f"{cfg.joint.thickness * 1e3:.0f} mm mild steel, seed {args.seed}"
    )
    runs = {}
    for name, ctl in controllers.items():
        t0 = time.time()
        res = simulate(cfg, seed=args.seed, controller=ctl)
        runs[name] = {
            "table": res.table,
            "metrics": score(res.table, cfg, name),
            "controller": ctl,
        }
        print(f"  {runs[name]['metrics'].headline()}   [{time.time() - t0:.1f} s]")

    args.out.mkdir(parents=True, exist_ok=True)
    fig = summary_figure(runs, cfg, args.seed, args.gap_profile)
    fig.savefig(args.out / "summary.png", dpi=140)
    plt.close(fig)
    fig = dashboard_figure(runs, cfg)
    fig.savefig(args.out / "dashboard.png", dpi=130)
    plt.close(fig)

    base, adap = runs["baseline"]["metrics"], runs["adaptive"]["metrics"]
    metrics = {
        "seed": args.seed,
        "gap_profile": args.gap_profile,
        "seam_length_mm": args.length * 1e3,
        "plate_thickness_mm": cfg.joint.thickness * 1e3,
        "residual_used": residual is not None,
        "wall_clock_s": None,  # filled below
        "controllers": {n: d["metrics"].as_dict() for n, d in runs.items()},
        "improvement": {
            "burn_through_length_ratio": (
                adap.burn_through_length_mm / base.burn_through_length_mm
                if base.burn_through_length_mm > 0 else None
            ),
            "penetration_std_ratio": adap.penetration_std_mm / base.penetration_std_mm,
            "cycle_time_ratio": adap.duration_s / base.duration_s,
            "in_band_gain_pct": adap.penetration_in_band_pct - base.penetration_in_band_pct,
        },
    }
    metrics["wall_clock_s"] = round(time.time() - t_start, 1)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print()
    print(f"  burn-through length : {base.burn_through_length_mm:6.1f} -> "
          f"{adap.burn_through_length_mm:5.1f} mm")
    print(f"  penetration std     : {base.penetration_std_mm:6.2f} -> "
          f"{adap.penetration_std_mm:5.2f} mm")
    print(f"  inside target band  : {base.penetration_in_band_pct:6.1f} -> "
          f"{adap.penetration_in_band_pct:5.1f} %")
    print(f"  cycle time          : {base.duration_s:6.1f} -> {adap.duration_s:5.1f} s")
    print()
    for f in ("summary.png", "dashboard.png", "metrics.json"):
        print(f"  wrote {args.out / f}")
    print(f"\ntotal {metrics['wall_clock_s']:.1f} s wall clock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
