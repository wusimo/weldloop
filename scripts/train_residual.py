#!/usr/bin/env python3
"""Train the optional melt-pool residual network.

Exits cleanly (status 0) with an explanatory message if torch is unavailable —
nothing in the demo depends on this script having been run.

Usage::

    python scripts/train_residual.py --n-seams 4 --out data/residual.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from weldloop.config import default_config
from weldloop.estimation.residual import ResidualDataset, build_dataset, torch_available, train
from weldloop.sim.runner import simulate

KINDS = ("step", "ramp", "sine", "random")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seams", type=int, default=4)
    ap.add_argument("--length", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=100, help="offset from the demo seeds")
    ap.add_argument("--out", type=Path, default=Path("data/residual.pt"))
    args = ap.parse_args()

    if not torch_available():
        print("torch is not installed; skipping residual training.")
        print("The demo runs on pure physics without it — this is not an error.")
        return 0

    parts = []
    for i in range(args.n_seams):
        kind = KINDS[i % len(KINDS)]
        seed = args.seed + i
        cfg = default_config(seam__kind=kind, seam__length=args.length, sim__seed=seed)
        table = simulate(cfg, seed=seed).table
        ds = build_dataset(cfg, table)
        parts.append(ds)
        print(f"[{i + 1}/{args.n_seams}] {kind:<8s} seed={seed}  {len(ds)} samples")

    data = ResidualDataset(
        X=np.concatenate([p.X for p in parts]),
        Y=np.concatenate([p.Y for p in parts]),
        dt=parts[0].dt,
    )
    print(f"training on {len(data)} samples")
    model, stats = train(data, epochs=args.epochs)
    path = model.save(args.out)
    print(f"\nsaved {path}")
    print(
        f"  one-step penetration model error: "
        f"{stats['rms_p_before_mm']:.4f} mm -> {stats['rms_p_after_mm']:.4f} mm "
        f"on held-out data"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
