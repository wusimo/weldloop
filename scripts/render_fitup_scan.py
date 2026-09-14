#!/usr/bin/env python3
"""Render the fit-up inspection report the planning layer is given.

    python scripts/render_fitup_scan.py

Writes ``docs/vla_fitup_scan.png``.  This image is an *input* to the demo, not
a result: it is the only place the task planner is told what the joint looks
like, so it has to be regenerated whenever the demo seam changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from weldloop.planning.job import DEMO_JOB, demo_config, demo_seam
from weldloop.viz.fitup import fitup_report_figure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("docs/vla_fitup_scan.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = demo_config(seed=args.seed)
    seam = demo_seam(cfg, seed=args.seed)
    fig = fitup_report_figure(seam, cfg, job_id=DEMO_JOB.job_id,
                              scan_date=DEMO_JOB.scan_date)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
