#!/usr/bin/env python3
"""Generate a dataset of simulated welds in the phase-1 capture schema.

Each run writes one wide table (Parquet, or gzipped CSV if pyarrow is absent)
plus a manifest describing the seam and what went wrong on it.  The point of
this script is not the data — it is that **the file layout, column names,
units and rates are the ones we are proposing to capture on the real cell**.
When real data arrives in this shape, `estimation/` and `control/` run on it
unchanged.

Usage::

    python scripts/make_dataset.py --n 6 --out data
    python scripts/make_dataset.py --n 2 --no-truth     # capture-shaped only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import numpy as np

from weldloop.config import default_config
from weldloop.sim.logger import SCHEMA, schema_markdown
from weldloop.sim.runner import simulate

KINDS = ("step", "ramp", "sine", "random", "constant")


def summarise(table, seam, cfg) -> dict:
    """Per-run metadata: what seam it was, and what the fixed parameters did to it."""
    s = table["rb_s"]
    ds = np.diff(s, prepend=s[0])
    out = {
        "seam_kind": seam.kind,
        "seam_length_m": float(seam.length),
        "gap_min_mm": float(seam.gap.min() * 1e3),
        "gap_max_mm": float(seam.gap.max() * 1e3),
        "duration_s": float(table["t"][-1]),
        "rows": int(table.n_rows),
        "f_master_hz": float(table.f_master),
    }
    if "truth_penetration" in table.columns:
        p = table["truth_penetration"]
        bt = table["truth_burn_through"] > 0.5
        lof = table["truth_lack_of_fusion"] > 0.5
        out.update(
            {
                "penetration_mean_mm": float(np.nanmean(p) * 1e3),
                "penetration_std_mm": float(np.nanstd(p) * 1e3),
                "burn_through_length_mm": float(np.nansum(ds[bt]) * 1e3),
                "lack_of_fusion_length_mm": float(np.nansum(ds[lof]) * 1e3),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6, help="number of welds")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length", type=float, default=0.20, help="seam length [m]")
    ap.add_argument(
        "--kinds", type=str, default=",".join(KINDS), help="comma-separated gap profiles"
    )
    ap.add_argument(
        "--no-truth",
        action="store_true",
        help="omit truth_* columns, producing a table shaped exactly like a real capture",
    )
    args = ap.parse_args()

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator": "weldloop scripts/make_dataset.py",
        "include_truth": not args.no_truth,
        "schema": [
            {
                "name": c.name, "unit": c.unit, "source": c.source,
                "rate_hz": c.rate_hz, "reducer": c.reducer, "real_hw": c.real_hw,
                "description": c.description,
            }
            for c in SCHEMA
        ],
        "runs": [],
    }

    t0 = time.time()
    for i in range(args.n):
        kind = kinds[i % len(kinds)]
        seed = args.seed + i
        cfg = default_config(seam__kind=kind, seam__length=args.length, sim__seed=seed)
        res = simulate(cfg, seed=seed, include_truth=not args.no_truth)
        path = res.table.write(args.out / f"weld_{i:03d}")
        entry = {"file": path.name, "seed": seed, **summarise(res.table, res.seam, cfg)}
        manifest["runs"].append(entry)
        print(
            f"[{i + 1}/{args.n}] {path.name}  {kind:<8s} "
            f"{entry['rows']:>7d} rows  {path.stat().st_size / 1e6:5.1f} MB"
            + (
                f"  burn-through {entry['burn_through_length_mm']:5.1f} mm"
                if "burn_through_length_mm" in entry
                else ""
            )
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "SCHEMA.md").write_text(
        "# weldloop capture schema\n\n"
        "本表即为一期真实采集建议 schema。仿真与真实数据使用同一列名、单位与速率。\n\n"
        + schema_markdown()
        + "\n"
    )
    print(
        f"\nwrote {args.n} runs + manifest.json + SCHEMA.md to {args.out}/ "
        f"in {time.time() - t0:.1f} s"
    )


if __name__ == "__main__":
    main()
