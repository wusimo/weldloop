#!/usr/bin/env python3
"""Run the R4 task planner on its own and print what it produced.

    python scripts/vla_plan.py                 # replay the recorded response
    python scripts/vla_plan.py --live          # call the API and re-record

The default path needs no key and no network: it replays the response saved
under ``data/vla/``, keyed by a digest of the exact request.  ``--live``
requires ``pip install anthropic`` and a credential, and overwrites the
recording for whatever request the current job card, prompt and scan produce.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weldloop.planning.job import (
    DEMO_JOB, FitupScan, demo_config, demo_seam, demo_seam_description,
)
from weldloop.planning.schedule import PlanSchedule
from weldloop.planning.vla import VLATaskPlanner, resolve_backend, save_recording


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="call the model instead of replaying the recording")
    ap.add_argument("--scan", type=Path, default=Path("docs/vla_fitup_scan.png"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="also dump the plan here")
    args = ap.parse_args()

    cfg = demo_config(seed=args.seed)
    seam = demo_seam(cfg, seed=args.seed)
    desc = demo_seam_description(cfg)
    scan = FitupScan(args.scan)
    if not scan.exists():
        print(f"! {args.scan} is missing — run scripts/render_fitup_scan.py first")
        return 1

    planner = VLATaskPlanner(cfg, backend=resolve_backend(prefer_live=args.live))
    plan, trace = planner.plan(DEMO_JOB, scan, desc, measured=seam)
    schedule = PlanSchedule.from_plan(plan, cfg)

    if args.live and not trace.fallback:
        path = save_recording(trace.raw, trace.digest, trace.model,
                              note="recorded by scripts/vla_plan.py --live")
        print(f"recorded -> {path}\n")

    print(f"backend {trace.backend} / {trace.model}   request {trace.digest}")
    print(f"latency {trace.latency_s:.2f} s "
          f"= {trace.cycles_missed(cfg):.0f} x the {cfg.control.dt * 1e3:.0f} ms "
          f"motion-layer period")
    if trace.fallback:
        print(f"! fell back to the rule planner: {trace.fallback}")
    print()
    if trace.reading:
        print("读图 reading:\n  " + trace.reading.replace("。", "。\n  ") + "\n")

    print(f"{'seg':>3} {'s [mm]':>14} {'class':<8} {'I [A]':>6} {'v [mm/s]':>9} "
          f"{'weave':>6} {'h [mm]':>7} {'band [mm]':>12}")
    for i, (seg, nom) in enumerate(zip(plan.segments, schedule.nominals)):
        print(f"{i:>3} {seg.s_start * 1e3:6.0f}-{seg.s_end * 1e3:<7.0f} "
              f"{seg.gap_class:<8} {nom.I_set:6.1f} {nom.v_travel * 1e3:9.2f} "
              f"{nom.weave_amp * 1e3:6.2f} {nom.thickness * 1e3:7.1f} "
              f"{nom.p_lo * 1e3:5.2f}-{nom.p_hi * 1e3:<5.2f}")
    for i, seg in enumerate(plan.segments):
        print(f"    [{i}] {seg.rationale}")

    if schedule.n_clamped:
        print(f"\n! {schedule.n_clamped} setpoint(s) clamped by the cell:")
        for line in schedule.clamp_lines():
            print(f"    {line}")
    else:
        print("\n  every setpoint was inside the machine and procedure limits")

    if trace.risks:
        print("\n风险 risks flagged by the planner:")
        for r in trace.risks:
            print(f"  - {r}")
    print(f"\nestimated cycle time {plan.estimated_cycle_time:.1f} s")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"plan": plan.as_dict(), "trace": trace.as_dict()},
                       indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
