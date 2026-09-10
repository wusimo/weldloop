"""Render the weld from the MuJoCo cell — real robot, real meshes, real shadows.

This is the same experiment as everywhere else in the repository, run with a
UR10e in the loop (``sim/mujoco_cell.py``) and drawn with MuJoCo's renderer
rather than with matplotlib primitives.  The arm, the torch, the bench and the
workpiece are geometry in a physics model; the weld bead is revealed segment by
segment as the torch passes; the arc is an emissive body carrying a spotlight,
so it actually lights the plate and casts shadows.

What is simulated and what is drawn
-----------------------------------
Simulated: the arm (link inertias, joint limits, PD position actuators,
gravity compensation), the TCP trajectory, and — through ``weldloop`` — the
whole weld.  Drawn: the bead colour, the arc glow radius and the light
intensity, all driven from the logged process state.

MuJoCo has no volumetric smoke, so fume is not attempted here; the Blender
render (``viz/blender_export.py``) is where that belongs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weldloop.config import WeldConfig
from weldloop.sim.mujoco_cell import CellLayout, MujocoRobot
from weldloop.sim.seam import Seam

__all__ = ["RunRecord", "record_run", "render_cell_video"]


@dataclass(slots=True)
class RunRecord:
    """One controller's run, resampled onto the render clock."""

    name: str
    t: np.ndarray
    qpos: np.ndarray          # (N, 6)
    s: np.ndarray
    gap: np.ndarray
    p: np.ndarray
    w: np.ndarray
    I: np.ndarray
    smoke: np.ndarray
    burn_through: np.ndarray
    v_travel: np.ndarray
    tracking_error: float
    metrics: object
    table: object


def record_run(
    cfg: WeldConfig,
    seam: Seam,
    controller,
    *,
    seed: int = 0,
    record_hz: float = 50.0,
    layout: CellLayout | None = None,
) -> RunRecord:
    """Run the weld with the UR10e in the loop and keep the joint trajectory."""
    from weldloop.metrics import score
    from weldloop.sim.runner import simulate

    robot = MujocoRobot(cfg, seam, layout=layout, record_hz=record_hz)
    res = simulate(cfg, seed=seed, seam=seam, controller=controller, robot=robot)
    table = res.table

    t = np.array([h[0] for h in robot.history])
    qpos = np.array([h[1] for h in robot.history])

    def resample(col: str) -> np.ndarray:
        return np.interp(t, table["t"], table[col])

    return RunRecord(
        name=getattr(controller, "name", "controller"),
        t=t,
        qpos=qpos,
        s=resample("rb_s"),
        gap=resample("truth_gap"),
        p=resample("truth_penetration"),
        w=resample("truth_pool_w"),
        I=resample("ps_I"),
        smoke=resample("truth_smoke"),
        burn_through=resample("truth_burn_through"),
        v_travel=resample("rb_v_travel"),
        tracking_error=float(robot.tracking_error),
        metrics=score(table, cfg, getattr(controller, "name", "controller")),
        table=table,
    )


# --------------------------------------------------------------------------
class _SceneDresser:
    """Drives the bead, the arc glow and the arc light from the process state."""

    def __init__(self, model, layout: CellLayout, seam: Seam) -> None:
        import mujoco

        self.model = model
        self.layout = layout
        self.seam = seam
        self._mj = mujoco
        self.bead_ids = []
        i = 0
        while True:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"bead_{i:03d}")
            if gid < 0:
                break
            self.bead_ids.append(gid)
            i += 1
        self.bead_s = np.array(
            [(j + 0.5) / len(self.bead_ids) * seam.length for j in range(len(self.bead_ids))]
        )
        # The bead boxes are axis-aligned: index along the seam is the seam
        # direction, the other horizontal index is the bead width.  Writing the
        # width to the wrong axis makes the bead render as confetti.
        across = np.cross([0.0, 0.0, 1.0], np.asarray(layout.seam_dir, dtype=float))
        self._width_axis = int(np.argmax(np.abs(across)))
        self.core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arc_core")
        self.halo = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arc_halo")
        self.arc_light = model.nlight - 1     # the arc light was added last
        self._base_rgba = model.geom_rgba.copy()

    def dress(self, data, rec: RunRecord, k: int, bt_mask: np.ndarray) -> None:
        m = self.model
        s_now = rec.s[k]
        laid = self.bead_s <= s_now
        age = np.clip((s_now - self.bead_s) / 0.012, 0.0, 1.0)     # ~12 mm to cool

        for j, gid in enumerate(self.bead_ids):
            if not laid[j]:
                m.geom_rgba[gid, 3] = 0.0
                continue
            a = age[j]
            if bt_mask[j]:
                m.geom_rgba[gid] = (0.62, 0.10, 0.09, 1.0)
            else:
                # cools from white-hot through orange to dark weld metal
                m.geom_rgba[gid] = (
                    0.30 + 0.70 * (1.0 - a) ** 1.3,
                    0.28 + 0.62 * (1.0 - a) ** 2.0,
                    0.27 + 0.42 * (1.0 - a) ** 3.2,
                    1.0,
                )
            # a wider pool leaves a wider bead
            m.geom_size[gid, self._width_axis] = max(0.42 * rec.w[k], 0.002)

        # the arc itself
        tip = self.layout.seam_point(s_now) + np.array([0.0, 0.0, 0.0015])
        data.mocap_pos[0] = tip
        duty = float(np.clip(rec.I[k] / 260.0, 0.2, 1.6))
        m.geom_size[self.core, 0] = 0.0038 + 0.0020 * duty
        m.geom_size[self.halo, 0] = 0.0055 + 0.0035 * duty
        m.geom_rgba[self.halo] = (1.0, 0.90, 0.72, 0.16 + 0.12 * duty)
        m.light_diffuse[self.arc_light] = np.array([1.0, 0.72, 0.40]) * duty
        m.light_specular[self.arc_light] = np.array([0.8, 0.6, 0.4]) * duty


def render_cell_video(
    records: dict[str, RunRecord],
    cfg: WeldConfig,
    seam: Seam,
    out_path,
    *,
    hero: str = "adaptive",
    width: int = 1280,
    height: int = 720,
    fps: int = 25,
    n_frames: int | None = None,
    progress: bool = True,
):
    """Composite video: the cell, a torch close-up, and the measured comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import mujoco

    from weldloop.viz.style import C, use_style

    use_style()
    rec = records[hero]
    other = records.get("baseline" if hero != "baseline" else "adaptive")

    robot = MujocoRobot(cfg, seam)          # a fresh model purely for rendering
    model, data = robot.model, robot.data
    dresser = _SceneDresser(model, robot.layout, seam)

    big = mujoco.Renderer(model, height, width)
    small = mujoco.Renderer(model, 360, 480)

    n = len(rec.t) if n_frames is None else min(n_frames, len(rec.t))
    idx = np.linspace(0, len(rec.t) - 1, n).astype(int)

    # which bead segments were laid while the flag was set
    bt_mask = np.interp(dresser.bead_s, rec.s, rec.burn_through) > 0.5

    fig = plt.figure(figsize=(15.4, 8.7))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[2.35, 1.0], height_ratios=[1.0, 0.42],
        left=0.028, right=0.978, top=0.915, bottom=0.075, hspace=0.20, wspace=0.05,
    )
    ax_big = fig.add_subplot(gs[0, 0])
    ax_small = fig.add_subplot(gs[0, 1])
    ax_strip = fig.add_subplot(gs[1, :])
    for ax in (ax_big, ax_small):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    im_big = ax_big.imshow(np.zeros((height, width, 3), np.uint8))
    im_small = ax_small.imshow(np.zeros((360, 480, 3), np.uint8))
    ax_big.set_title(
        "MuJoCo 单元：UR10e + 焊枪 + 夹具（机器人在环，TCP 由逆运动学 + 位置伺服实现）",
        loc="left", fontsize=11,
    )
    ax_small.set_title("焊枪特写", loc="left", fontsize=11)

    hud = ax_big.text(
        0.011, 0.975, "", transform=ax_big.transAxes, color="#ffffff", fontsize=11,
        va="top", linespacing=1.5,
        bbox=dict(facecolor="#0d1013", edgecolor="#3a4048", alpha=0.82, pad=6),
    )
    warn = ax_big.text(
        0.011, 0.035, "", transform=ax_big.transAxes, color="#ffb3b3", fontsize=12,
        va="bottom", bbox=dict(facecolor="#3a1416", edgecolor=C.bad, alpha=0.92, pad=5),
    )

    if other is not None:
        ax_strip.plot(other.s * 1e3, other.p * 1e3, color=C.baseline, lw=1.4,
                      label=f"定参数 baseline（烧穿 {other.metrics.burn_through_length_mm:.1f} mm）")
    ax_strip.plot(rec.s * 1e3, rec.p * 1e3, color=C.adaptive, lw=1.6,
                  label=f"自适应 adaptive（烧穿 {rec.metrics.burn_through_length_mm:.1f} mm）")
    ax_strip.axhspan(cfg.control.p_lo * 1e3, cfg.control.p_hi * 1e3,
                     color=C.band_ok, alpha=0.13, lw=0)
    ax_strip.axhline(cfg.joint.thickness * 1e3, color=C.bad, ls="--", lw=1.1,
                     label="板厚 = 烧穿")
    ax_strip.set_ylim(2.2, 6.6)
    ax_strip.set_xlim(0.0, seam.length * 1e3)
    ax_strip.set_ylabel("真实熔深 [mm]", fontsize=9)
    ax_strip.set_xlabel("沿焊缝位置 s [mm]", fontsize=9)
    ax_strip.legend(loc="lower right", ncols=3, fontsize=8.5)
    (cursor,) = ax_strip.plot([], [], color=C.text, lw=1.2, alpha=0.9)

    fig.suptitle(
        "weldloop × MuJoCo —— 电源在环自适应焊接，UR10e 在环   "
        "（机械臂动力学、关节限位与位置伺服均为仿真；焊接物理与其余图表同源）",
        color=C.text, fontsize=12.5, fontweight="bold", y=0.982,
    )

    def draw(f: int):
        k = int(idx[f])
        data.qpos[:6] = rec.qpos[k]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        dresser.dress(data, rec, k, bt_mask)

        big.update_scene(data, camera="cell")
        im_big.set_data(big.render())
        small.update_scene(data, camera="closeup")
        im_small.set_data(small.render())

        hud.set_text(
            f"t = {rec.t[k]:5.1f} s\n"
            f"v = {rec.v_travel[k] * 1e3:5.2f} mm/s\n"
            f"I = {rec.I[k]:5.0f} A\n"
            f"间隙 = {rec.gap[k] * 1e3:4.2f} mm\n"
            f"熔深 = {rec.p[k] * 1e3:4.2f} / {cfg.joint.thickness * 1e3:.0f} mm"
        )
        msgs = []
        if rec.burn_through[k] > 0.5:
            msgs.append("本机烧穿")
        if other is not None and np.interp(rec.s[k], other.s, other.burn_through) > 0.5:
            msgs.append("定参数在此处已烧穿")
        warn.set_text("   ".join(msgs))
        warn.set_visible(bool(msgs))

        x = rec.s[k] * 1e3
        cursor.set_data([x, x], [2.2, 6.6])
        if progress and f % 25 == 0:
            print(f"    frame {f}/{n}", flush=True)
        return ()

    anim = animation.FuncAnimation(fig, draw, frames=n, blit=False)
    out_path = str(out_path)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=6000)
        anim.save(out_path, writer=writer, dpi=100,
                  savefig_kwargs={"facecolor": C.bg})
    except (FileNotFoundError, RuntimeError):
        out_path = out_path.rsplit(".", 1)[0] + ".gif"
        anim.save(out_path, writer=animation.PillowWriter(fps=min(fps, 12)), dpi=100,
                  savefig_kwargs={"facecolor": C.bg})
    plt.close(fig)
    big.close()
    small.close()
    return out_path
