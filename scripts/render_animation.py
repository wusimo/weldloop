#!/usr/bin/env python3
"""Render the side-by-side weld animation.

    python scripts/render_animation.py --seed 0 --gap-profile step

Writes ``out/weldloop.mp4`` (or a GIF if ffmpeg is unavailable).  The two
controllers are played against **position along the seam**, not time, so the
comparison stays aligned even though they take different numbers of seconds;
each arm carries its own clock so the cycle-time difference is visible.

The render is a visualisation of the reduced-order physics used everywhere
else in the repo — analytic temperature field, ROM pool cross-section,
procedural smoke — and the figure says so.  It is not CFD.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weldloop.config import default_config
from weldloop.control.baseline import BaselineController
from weldloop.control.motion_layer import AdaptiveController
from weldloop.estimation.residual import load_residual
from weldloop.metrics import score
from weldloop.sim.runner import simulate
from weldloop.viz.render import render_animation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-profile", default="step")
    ap.add_argument("--length", type=float, default=0.20)
    ap.add_argument("--frames", type=int, default=420)
    ap.add_argument("--fps", type=int, default=21)
    ap.add_argument("--dpi", type=int, default=108)
    ap.add_argument("--out", type=Path, default=Path("out/weldloop.mp4"))
    ap.add_argument("--residual", type=Path, default=Path("data/residual.pt"))
    ap.add_argument("--no-residual", action="store_true")
    args = ap.parse_args()

    cfg = default_config(
        seam__kind=args.gap_profile, seam__length=args.length, sim__seed=args.seed
    )
    residual = None if args.no_residual else load_residual(args.residual)

    t0 = time.time()
    runs = {}
    for name, ctl in (
        ("baseline", BaselineController(cfg)),
        ("adaptive", AdaptiveController(cfg, residual=residual)),
    ):
        res = simulate(cfg, seed=args.seed, controller=ctl)
        runs[name] = {"table": res.table, "controller": ctl,
                      "metrics": score(res.table, cfg, name)}
        print(f"  {runs[name]['metrics'].headline()}")
    seam = res.seam
    print(f"  simulation {time.time() - t0:.1f} s; rendering {args.frames} frames...")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    path = render_animation(
        runs, cfg, seam, args.out,
        n_frames=args.frames, fps=args.fps, dpi=args.dpi,
    )
    print(f"\nwrote {path}  ({Path(path).stat().st_size / 1e6:.1f} MB, "
          f"render {time.time() - t1:.1f} s, total {time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
