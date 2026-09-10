#!/usr/bin/env python3
"""Render the welding cell from MuJoCo: real robot, real meshes, real shadows.

    python scripts/render_mujoco.py --seed 0 --gap-profile step

Runs the weld twice with a UR10e in the loop and writes
``out/weldloop_mujoco.mp4``: the cell, a torch close-up, and both controllers'
penetration underneath.  Faster than the Blender pipeline and good enough for
day-to-day work; ``scripts/render_photoreal.py`` is the presentation version.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weldloop.config import default_config
from weldloop.control.baseline import BaselineController
from weldloop.control.motion_layer import AdaptiveController
from weldloop.estimation.residual import load_residual
from weldloop.sim.seam import make_seam
from weldloop.viz.mujoco_render import record_run, render_cell_video


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-profile", default="step")
    ap.add_argument("--length", type=float, default=0.20)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--out", type=Path, default=Path("out/weldloop_mujoco.mp4"))
    ap.add_argument("--no-residual", action="store_true")
    args = ap.parse_args()

    cfg = default_config(
        seam__kind=args.gap_profile, seam__length=args.length, sim__seed=args.seed
    )
    seam = make_seam(cfg.seam, args.seed)
    residual = None if args.no_residual else load_residual(Path("data/residual.pt"))

    t0 = time.time()
    records = {
        "baseline": record_run(cfg, seam, BaselineController(cfg), seed=args.seed),
        "adaptive": record_run(
            cfg, seam, AdaptiveController(cfg, residual=residual), seed=args.seed
        ),
    }
    for r in records.values():
        print(f"  {r.metrics.headline()}   TCP err {r.tracking_error * 1e3:.3f} mm")
    print(f"  simulation {time.time() - t0:.1f} s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    path = render_cell_video(
        records, cfg, seam, args.out,
        width=args.width, height=int(args.width * 9 / 16),
        fps=args.fps, n_frames=args.frames,
    )
    print(f"\nwrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB, "
          f"render {time.time() - t1:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
