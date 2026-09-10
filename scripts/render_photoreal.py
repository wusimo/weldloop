#!/usr/bin/env python3
"""End-to-end photoreal pipeline: weldloop -> MuJoCo -> Blender -> MP4.

    python scripts/render_photoreal.py --stage all

Stages, each runnable on its own:

``sim``
    Run the weld twice (fixed parameters and adaptive) with a UR10e in the
    loop, and export the joint trajectory plus the process state.
``blender``
    Shell out to Blender twice — a wide cell shot and a torch close-up — and
    render PNG sequences with Cycles.
``composite``
    Overlay the live process numbers and the measured penetration comparison,
    and encode the two shots into one MP4.

The division of labour is strict: weldloop owns the physics, MuJoCo owns the
kinematics and the arm dynamics, Blender owns only the pixels.  Every number
burnt into the frames comes from the simulation log.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

BLENDER = Path("/home/simo/Downloads/blender-5.2.0-linux-x64/blender")
LOOK = [
    "--bead-strength", "220", "--cool-len", "0.055",
    "--arc-energy", "0.6", "--arc-emission", "4200",
    "--plate-rough", "0.80", "--plate-metal", "0.45",
    "--fume-density", "300", "--key", "240", "--rim", "70",
]


def stage_sim(args) -> None:
    from weldloop.config import default_config
    from weldloop.control.baseline import BaselineController
    from weldloop.control.motion_layer import AdaptiveController
    from weldloop.estimation.residual import load_residual
    from weldloop.sim.mujoco_cell import MujocoRobot
    from weldloop.sim.seam import make_seam
    from weldloop.viz.blender_export import export_scene
    from weldloop.viz.mujoco_render import record_run

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
    robot = MujocoRobot(cfg, seam)
    export_scene(records, cfg, seam, robot.layout, robot.model,
                 args.scene, n_frames=args.frames)
    print(f"  exported {args.frames} frames to {args.scene} in {time.time() - t0:.1f} s")


def stage_blender(args) -> None:
    if not BLENDER.exists():
        raise SystemExit(f"Blender not found at {BLENDER}; pass --blender")
    for view, out, exposure in (
        ("wide", args.out / "blender_wide", "-1.0"),
        ("close", args.out / "blender_close", "-3.0"),
    ):
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(BLENDER), "-b", "-noaudio", "-P",
            str(Path(__file__).with_name("blender_render.py")), "--",
            "--scene", str(args.scene), "--out", str(out), "--view", view,
            "--exposure", exposure, "--res", str(args.res),
            "--samples", str(args.samples), "--frames", str(args.frames), *LOOK,
        ]
        print(f"  rendering {view} -> {out}")
        t0 = time.time()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print(f"  {view} done in {time.time() - t0:.1f} s")


def stage_composite(args) -> None:
    import matplotlib.animation as animation
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    from weldloop.viz.style import C, use_style

    meta = json.loads((args.scene / "scene.json").read_text())
    fr = np.load(args.scene / "frames.npz")
    shots = []
    for view in ("wide", "close"):
        files = sorted((args.out / f"blender_{view}").glob("frame_*.png"))
        if files:
            shots.append((view, files))
    if not shots:
        raise SystemExit("no rendered frames found; run --stage blender first")

    use_style()
    first = mpimg.imread(shots[0][1][0])
    h, w = first.shape[:2]
    # Size the figure so the render fills its axes exactly: imshow keeps the
    # image aspect, so a mismatched axes just adds black bars.
    fig_w = 12.8
    img_frac_w, img_frac_h = 0.945, 0.700
    img_axes_w = fig_w * img_frac_w
    fig_h = (img_axes_w * h / w) / img_frac_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_img = fig.add_axes((0.0275, 0.255, img_frac_w, img_frac_h))
    ax_strip = fig.add_axes((0.055, 0.075, 0.905, 0.150))
    ax_img.set_xticks([]); ax_img.set_yticks([]); ax_img.grid(False)
    for sp in ax_img.spines.values():
        sp.set_color(C.grid)
    im = ax_img.imshow(first)

    hud = ax_img.text(
        0.012, 0.972, "", transform=ax_img.transAxes, color="#ffffff", fontsize=11,
        va="top", linespacing=1.55,
        bbox=dict(facecolor="#0b0d10", edgecolor="#39404a", alpha=0.80, pad=6),
    )
    warn = ax_img.text(
        0.012, 0.035, "", transform=ax_img.transAxes, color="#ffc2c2", fontsize=12,
        va="bottom", bbox=dict(facecolor="#3a1416", edgecolor="#e5484d",
                               alpha=0.92, pad=5),
    )
    shot_lbl = ax_img.text(
        0.988, 0.972, "", transform=ax_img.transAxes, color=C.muted, fontsize=10,
        va="top", ha="right",
    )

    s_mm = fr["s"] * 1e3
    ax_strip.plot(fr["other_s"] * 1e3, fr["other_p"] * 1e3, color=C.baseline, lw=1.4,
                  label=f"定参数 baseline（烧穿 "
                        f"{meta['metrics']['baseline']['burn_through_length_mm']:.1f} mm）")
    ax_strip.plot(s_mm, fr["p"] * 1e3, color=C.adaptive, lw=1.6,
                  label=f"自适应 adaptive（烧穿 "
                        f"{meta['metrics']['adaptive']['burn_through_length_mm']:.1f} mm）")
    ax_strip.axhspan(meta["p_lo_mm"], meta["p_hi_mm"], color=C.band_ok, alpha=0.13, lw=0)
    ax_strip.axhline(meta["plate_thickness_mm"], color=C.bad, ls="--", lw=1.1,
                     label="板厚 = 烧穿")
    ax_strip.set_ylim(2.2, 6.6)
    ax_strip.set_xlim(0.0, meta["seam_length"] * 1e3)
    ax_strip.set_ylabel("真实熔深 [mm]", fontsize=9)
    ax_strip.set_xlabel("沿焊缝位置 s [mm]", fontsize=9)
    ax_strip.legend(loc="lower right", ncols=3, fontsize=8.5)
    (cursor,) = ax_strip.plot([], [], color=C.text, lw=1.2, alpha=0.9)

    fig.suptitle(
        "weldloop —— 电源在环自适应焊接 | 焊接物理 weldloop · 机械臂 MuJoCo(UR10e) · 渲染 Blender Cycles\n"
        "画面中的每个数字都来自仿真日志；机械臂为动力学仿真，渲染不参与物理计算",
        color=C.text, fontsize=11.5, fontweight="bold", y=0.988,
    )

    order = [(v, i, f) for v, files in shots for i, f in enumerate(files)]
    n_total = len(order)

    def draw(idx: int):
        view, k, path = order[idx]
        im.set_data(mpimg.imread(path))
        k = min(k, len(fr["t"]) - 1)
        hud.set_text(
            f"t = {fr['t'][k]:5.1f} s\n"
            f"v = {fr['v_travel'][k] * 1e3:5.2f} mm/s\n"
            f"I = {fr['I'][k]:5.0f} A\n"
            f"间隙 = {fr['gap'][k] * 1e3:4.2f} mm\n"
            f"熔深 = {fr['p'][k] * 1e3:4.2f} / {meta['plate_thickness_mm']:.0f} mm"
        )
        msgs = []
        if fr["burn_through"][k] > 0.5:
            msgs.append("本机烧穿")
        if "other_bt" in fr and fr["other_bt"][k] > 0.5:
            msgs.append("定参数在此处已烧穿")
        warn.set_text("   ".join(msgs))
        warn.set_visible(bool(msgs))
        shot_lbl.set_text("单元全景" if view == "wide" else "焊枪特写")
        x = fr["s"][k] * 1e3
        cursor.set_data([x, x], [2.2, 6.6])
        if idx % 25 == 0:
            print(f"    composite {idx}/{n_total}", flush=True)
        return ()

    anim = animation.FuncAnimation(fig, draw, frames=n_total, blit=False)
    out_path = args.out / "weldloop_photoreal.mp4"
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=8000)
    anim.save(str(out_path), writer=writer, dpi=100,
              savefig_kwargs={"facecolor": C.bg})
    plt.close(fig)
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{n_total} frames)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("all", "sim", "blender", "composite"),
                    default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-profile", default="step")
    ap.add_argument("--length", type=float, default=0.20)
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--res", type=int, default=960)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--scene", type=Path, default=Path("out/blender_scene"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--no-residual", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "sim"):
        stage_sim(args)
    if args.stage in ("all", "blender"):
        stage_blender(args)
    if args.stage in ("all", "composite"):
        stage_composite(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
