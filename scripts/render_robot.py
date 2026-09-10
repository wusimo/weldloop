#!/usr/bin/env python3
"""Render the third-person view: a robot arm welding the variable-gap seam.

    python scripts/render_robot.py --seed 0 --gap-profile step

Writes ``out/weldloop_robot.mp4``.  The arm is a kinematic visualisation built
around the tool-centre-point trajectory the simulation already produces — it
follows real commanded motion, and it does not feed back into the physics.
The inverse kinematics is solved over the whole seam first, so the run also
reports whether the commanded path is reachable inside the joint limits.
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

from weldloop.config import default_config
from weldloop.control.baseline import BaselineController
from weldloop.control.motion_layer import AdaptiveController
from weldloop.estimation.residual import load_residual
from weldloop.metrics import score
from weldloop.sim.runner import simulate
from weldloop.viz.scene3d import render_robot_scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-profile", default="step")
    ap.add_argument("--length", type=float, default=0.20)
    ap.add_argument("--frames", type=int, default=360)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=104)
    ap.add_argument("--arm", choices=("adaptive", "baseline"), default="adaptive",
                    help="which controller the robot in the scene is running")
    ap.add_argument("--orbit", type=float, default=26.0, help="camera sweep [deg]")
    ap.add_argument("--out", type=Path, default=Path("out/weldloop_robot.mp4"))
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
    print(f"  simulation {time.time() - t0:.1f} s; rendering {args.frames} frames...")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    path, report = render_robot_scene(
        runs, cfg, res.seam, args.out,
        n_frames=args.frames, fps=args.fps, dpi=args.dpi,
        arm=args.arm, orbit=args.orbit,
    )
    (args.out.parent / "kinematics.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}  ({Path(path).stat().st_size / 1e6:.1f} MB, "
          f"render {time.time() - t1:.1f} s)")
    print(f"wrote {args.out.parent / 'kinematics.json'}: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
