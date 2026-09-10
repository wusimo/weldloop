"""Third-person render of the welding cell.

A robot arm carrying the torch along the seam, the plate glowing with the
analytic temperature field, fume rising off the arc, and the two camera views
beside it.  Underneath, both controllers' penetration against seam position,
so the intuitive picture and the measured result are on the same screen.

What is real and what is dressing
---------------------------------
* **Real**: the torch position, travel speed, weave, arc current and voltage,
  pool width and penetration, smoke density and every defect flag come from
  the simulation log.  The plate's surface temperature is Rosenthal's analytic
  field for the actual instantaneous power and travel speed.  The joint angles
  come from inverse kinematics of the commanded TCP path, checked against
  joint limits.
* **Dressing**: the arm's link lengths, the shape of the torch, the fume
  particles and the sparks.  None of it feeds back into the process model —
  the weld would be identical with no arm drawn at all.

The banner on the figure says this, because a render that implies more
fidelity than it has is worse than no render.
"""

from __future__ import annotations

import math

import numpy as np

from weldloop.config import WeldConfig
from weldloop.physics.heat_source import rosenthal_thick_plate
from weldloop.viz.render import RenderData, _CrossSection, _camera_frames, prepare
from weldloop.viz.robot import ArmGeometry, fk, solve_path, tcp_pose
from weldloop.viz.style import C, use_style

__all__ = ["render_robot_scene"]

PLATE_HALF_Y = 0.058
PLATE_MARGIN = 0.022
NX_PLATE, NY_PLATE = 120, 30

#: cold steel, and the temperature window over which it glows
STEEL_RGB = np.array([0.400, 0.435, 0.482])
GLOW_LO, GLOW_HI = 680.0, 1950.0


def _plate_grid(length: float):
    x = np.linspace(-PLATE_MARGIN, length + PLATE_MARGIN, NX_PLATE)
    y = np.linspace(-PLATE_HALF_Y, PLATE_HALF_Y, NY_PLATE)
    return np.meshgrid(x, y, indexing="ij")


def _fume_particles(rng, n: int = 150):
    """Static particle seeds; each frame maps them through the plume shape."""
    return (
        rng.random(n),                     # age phase
        rng.normal(0.0, 1.0, n),           # lateral spread
        rng.normal(0.0, 1.0, n),           # vertical wobble
        rng.uniform(0.6, 1.6, n),          # size
    )


def render_robot_scene(
    runs: dict,
    cfg: WeldConfig,
    seam,
    out_path,
    *,
    n_frames: int = 360,
    fps: int = 20,
    dpi: int = 104,
    arm: str = "adaptive",
    orbit: float = 26.0,
    progress: bool = True,
):
    """Render the third-person animation.  Returns the written path."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    use_style()
    s_grid, data = prepare(runs, cfg, n_frames)
    hero: RenderData = data[arm]
    other_name = "baseline" if arm != "baseline" else "adaptive"
    other: RenderData | None = data.get(other_name)
    length = float(s_grid[-1])

    # ---- kinematics over the whole path, once -------------------------
    geom = ArmGeometry()
    ctwd = cfg.robot.ctwd_nom
    pts, rots = [], []
    for k in range(n_frames):
        p, R = tcp_pose(
            float(hero.s[k]), float(hero.weave[k]), ctwd, drag_angle=0.26
        )
        pts.append(p)
        rots.append(R)
    pts = np.asarray(pts)
    Q, report = solve_path(pts, rots, geom)
    if progress:
        print(
            f"    kinematics: {report['unreachable_poses']} unreachable poses, "
            f"worst joint margin {math.degrees(report['worst_joint_margin_rad']):.1f} deg "
            f"({report['worst_joint']})"
        )

    # ---- figure --------------------------------------------------------
    fig = plt.figure(figsize=(14.0, 8.3))
    gs = fig.add_gridspec(
        4, 2, width_ratios=[1.72, 1.0], height_ratios=[1.0, 1.0, 1.0, 0.58],
        left=0.068, right=0.975, top=0.905, bottom=0.085, hspace=0.42, wspace=0.10,
    )
    ax3d = fig.add_subplot(gs[0:3, 0], projection="3d")
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_ir = fig.add_subplot(gs[1, 1])
    ax_sec = fig.add_subplot(gs[2, 1])
    ax_pen = fig.add_subplot(gs[3, :])
    section = _CrossSection(ax_sec, cfg, C.adaptive)

    fig.patch.set_facecolor(C.bg)
    ax3d.set_facecolor(C.bg)
    ax3d.set_axis_off()
    xlim = (-0.075, length + 0.075)
    ylim = (-0.505, 0.105)
    zlim = (-0.02, 0.53)
    try:
        ax3d.set_box_aspect(
            (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
        )
        ax3d.computed_zorder = False
    except AttributeError:  # pragma: no cover - very old matplotlib
        pass

    X, Y = _plate_grid(length)
    norm = colors.Normalize(vmin=300.0, vmax=2400.0)
    cmap = cm.get_cmap("inferno") if hasattr(cm, "get_cmap") else plt.get_cmap("inferno")

    # a table to stand everything on, so the scene reads as a cell
    tx = np.linspace(xlim[0] - 0.01, xlim[1] + 0.01, 2)
    ty = np.linspace(-PLATE_HALF_Y - 0.05, PLATE_HALF_Y + 0.05, 2)
    TX, TY = np.meshgrid(tx, ty, indexing="ij")
    ax3d.plot_surface(
        TX, TY, np.full_like(TX, -0.010), color="#262b31", shade=False,
        linewidth=0, antialiased=False, zorder=0,
    )
    geom_base = np.array([geom.base_xy[0], geom.base_xy[1], 0.0])
    ax3d.plot(
        [geom_base[0], geom_base[0]], [geom_base[1], geom_base[1]],
        [-0.012, geom.base_height * 0.55], color="#8b949e", lw=12,
        solid_capstyle="butt", zorder=5,
    )

    rng = np.random.default_rng(11)
    fume = _fume_particles(rng)
    state: dict = {"plate": None, "arm": None, "bead": None, "fume": None,
                   "arc": None, "torch": None, "spark": None}

    hud = ax3d.text2D(
        0.012, 0.965, "", transform=ax3d.transAxes, color=C.text, fontsize=11,
        va="top", linespacing=1.5,
        bbox=dict(facecolor=C.bg, edgecolor=C.grid, alpha=0.85, pad=5),
    )
    warn = ax3d.text2D(
        0.012, 0.055, "", transform=ax3d.transAxes, color=C.bad, fontsize=12,
        va="bottom",
        bbox=dict(facecolor="#3a1416", edgecolor=C.bad, alpha=0.9, pad=5),
    )

    # ---- camera panels --------------------------------------------------
    rgb0, ir0, _ = _camera_frames(hero, 0, cfg, rng)
    im_rgb = ax_rgb.imshow(rgb0, origin="lower", aspect="auto")
    im_ir = ax_ir.imshow(ir0, origin="lower", aspect="auto", cmap="inferno",
                         vmin=300.0, vmax=2400.0)
    for ax, title in ((ax_rgb, "RGB 相机（过程中不可用）"), (ax_ir, "IR 热像仪")):  # noqa: E501
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(title, fontsize=10, loc="left")
    txt_rgb = ax_rgb.text(
        0.03, 0.05, "", transform=ax_rgb.transAxes, color="#ffffff", fontsize=9.5,
        bbox=dict(facecolor="#000000", alpha=0.45, pad=2, edgecolor="none"),
    )

    # ---- penetration strip ---------------------------------------------
    if other is not None:
        ax_pen.plot(other.s * 1e3, other.p * 1e3, color=C.baseline, lw=1.3,
                    label="定参数 baseline")
    ax_pen.plot(hero.s * 1e3, hero.p * 1e3, color=C.adaptive, lw=1.5,
                label="自适应 adaptive（画面中这台）")
    ax_pen.axhspan(cfg.control.p_lo * 1e3, cfg.control.p_hi * 1e3,
                   color=C.band_ok, alpha=0.13, lw=0)
    ax_pen.axhline(cfg.joint.thickness * 1e3, color=C.bad, ls="--", lw=1.1,
                   label="板厚 = 烧穿")
    ax_pen.set_ylim(2.2, 6.6)
    ax_pen.set_xlim(0.0, length * 1e3)
    ax_pen.set_ylabel("真实熔深 [mm]", fontsize=9)
    ax_pen.set_xlabel("沿焊缝位置 s [mm]", fontsize=9)
    ax_pen.legend(loc="lower right", ncols=3, fontsize=8)
    (cursor,) = ax_pen.plot([], [], color=C.text, lw=1.2, alpha=0.85)

    fig.suptitle(
        "weldloop —— 电源在环自适应焊接：第三人称视角\n"
        "焊枪轨迹/热场/熔深/烟尘均来自仿真日志；机械臂为按该轨迹反解的运动学可视化，"
        "不参与物理计算",
        color=C.text, fontsize=12, fontweight="bold", y=0.985,
    )

    def draw(k: int):
        for key, art in state.items():
            if art is None:
                continue
            if isinstance(art, list):
                for a in art:
                    a.remove()
            else:
                art.remove()
            state[key] = None

        # --- plate: cold steel that glows where the analytic field is hot ---
        Q_arc = cfg.pool.eta_arc * hero.V[k] * hero.I[k]
        v = max(hero.v_travel[k], 1e-4)
        T = rosenthal_thick_plate(
            X - hero.s[k], Y - hero.weave[k], 0.0, Q_arc, v, cfg.material
        )
        # Compositing the glow over steel, rather than colour-mapping the field
        # directly, is what makes it read as hot metal instead of a heat map.
        glow = np.clip((T - GLOW_LO) / (GLOW_HI - GLOW_LO), 0.0, 1.0)[..., None]
        hot = cmap(norm(np.clip(T, 300.0, 2400.0)))[..., :3]
        rgb = STEEL_RGB[None, None, :] * (1.0 - glow) + hot * glow
        facecolors = np.concatenate([rgb, np.ones_like(glow)], axis=-1)

        # the still-open root gap ahead of the arc reads as a dark slot
        gap_here = np.asarray(seam.gap_at(np.clip(X, 0.0, seam.length)))
        ahead = X > hero.s[k]
        slot = (np.abs(Y) < np.maximum(gap_here, 3.0e-4) / 2.0) & ahead
        facecolors[slot] = np.array([0.04, 0.045, 0.05, 1.0])
        state["plate"] = ax3d.plot_surface(
            X, Y, np.zeros_like(X), facecolors=facecolors,
            rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False,
            zorder=1,
        )

        # --- solidified bead behind the torch, cooling as it recedes -------
        behind = s_grid <= hero.s[k]
        if behind.sum() > 3:
            sb = s_grid[behind]
            yb = np.interp(sb, hero.s, hero.weave)
            n_seg = 7
            edges = np.linspace(0, len(sb), n_seg + 1).astype(int)
            beads = []
            for j in range(n_seg):
                a, b = edges[j], min(edges[j + 1] + 1, len(sb))
                if b - a < 2:
                    continue
                # newest segment glows, older ones darken to weld-metal brown
                f = j / max(n_seg - 1, 1)
                col = (0.62 + 0.38 * f, 0.34 + 0.36 * f, 0.20 + 0.22 * f)
                beads.append(
                    ax3d.plot(sb[a:b], yb[a:b], np.full(b - a, 0.0015),
                              color=col, lw=3.4, solid_capstyle="round",
                              alpha=0.95, zorder=3)[0]
                )
            state["bead"] = beads

        # --- the arm ------------------------------------------------------
        joints, R = fk(Q[k], geom)
        state["arm"] = [
            ax3d.plot(joints[:4, 0], joints[:4, 1], joints[:4, 2],
                      color="#c9d1d9", lw=9.0, solid_capstyle="round", zorder=6)[0],
            ax3d.plot(joints[:4, 0], joints[:4, 1], joints[:4, 2], "o",
                      color="#7d8590", ms=10.0, zorder=7)[0],
            ax3d.plot([joints[3, 0], joints[4, 0]], [joints[3, 1], joints[4, 1]],
                      [joints[3, 2], joints[4, 2]], color=C.adaptive, lw=6.5,
                      solid_capstyle="round", zorder=7)[0],
        ]
        # torch nozzle and the wire stick-out
        mid = joints[3] + (joints[4] - joints[3]) * 0.62
        tip = joints[4] + R[:, 2] * 0.004
        state["torch"] = [
            ax3d.plot([joints[3, 0], mid[0]], [joints[3, 1], mid[1]],
                      [joints[3, 2], mid[2]], color="#4c5561", lw=11.0,
                      solid_capstyle="round", zorder=7)[0],
            ax3d.plot([mid[0], joints[4, 0]], [mid[1], joints[4, 1]],
                      [mid[2], joints[4, 2]], color="#9aa4b1", lw=6.0,
                      solid_capstyle="round", zorder=8)[0],
            ax3d.plot([joints[4, 0], tip[0]], [joints[4, 1], tip[1]],
                      [joints[4, 2], tip[2]], color="#ffd7a0", lw=2.4, zorder=9)[0],
        ]

        # --- arc glow -----------------------------------------------------
        arc_pt = np.array([hero.s[k], hero.weave[k], 0.0015])
        glow = []
        for size, alpha in ((520.0, 0.16), (240.0, 0.30), (90.0, 0.75), (26.0, 1.0)):
            glow.append(
                ax3d.scatter(
                    [arc_pt[0]], [arc_pt[1]], [arc_pt[2]], s=size, alpha=alpha,
                    color="#fff3d0", edgecolors="none", zorder=9,
                )
            )
        state["arc"] = glow

        # --- fume ---------------------------------------------------------
        phase, lateral, wobble, size = fume
        age = (phase + k / 42.0) % 1.0
        fx = hero.s[k] - 0.055 * age + 0.004 * wobble * age
        fy = hero.weave[k] + 0.030 * lateral * age
        fz = 0.004 + 0.16 * age ** 0.85
        keep = age < 0.98
        alpha_f = np.clip(0.55 * hero.smoke[k] * (1.0 - age), 0.0, 0.55)
        state["fume"] = [
            ax3d.scatter(
                fx[keep], fy[keep], fz[keep],
                s=(14.0 + 130.0 * age[keep]) * size[keep],
                c="#d8d5cf", alpha=float(alpha_f.mean()), edgecolors="none",
                zorder=8,
            )
        ]

        # --- sparks -------------------------------------------------------
        srng = np.random.default_rng(1000 + k)
        m = 9
        ang = srng.uniform(0.0, 2 * math.pi, m)
        rad = srng.uniform(0.004, 0.030, m)
        state["spark"] = [
            ax3d.scatter(
                arc_pt[0] + rad * np.cos(ang), arc_pt[1] + rad * np.sin(ang),
                0.001 + srng.uniform(0.0, 0.02, m), s=srng.uniform(2.0, 11.0, m),
                c="#ffcf6b", alpha=0.85, edgecolors="none", zorder=9,
            )
        ]

        # --- cross-section, cameras, HUD, cursor --------------------------
        section.update(hero, k)
        rgb, ir, quality = _camera_frames(hero, k, cfg, rng)
        im_rgb.set_data(rgb)
        im_ir.set_data(ir)
        usable = quality >= cfg.sensors.rgb_quality_min
        txt_rgb.set_text(f"可用度 {quality:.3f}" + ("" if usable else "   不可用"))
        txt_rgb.set_color("#ffffff" if usable else "#ff7b7b")

        hud.set_text(
            f"t = {hero.t[k]:5.1f} s\n"
            f"v = {hero.v_travel[k] * 1e3:5.2f} mm/s\n"
            f"I = {hero.I_cmd[k]:5.0f} A\n"
            f"间隙 = {hero.gap[k] * 1e3:4.2f} mm\n"
            f"熔深 = {hero.p[k] * 1e3:4.2f} mm"
        )
        msgs = []
        if hero.burn_through[k] > 0.5:
            msgs.append("本机烧穿")
        if other is not None and other.burn_through[k] > 0.5:
            msgs.append(f"定参数在此处已烧穿（{other.p[k] * 1e3:.2f} mm / "
                        f"{cfg.joint.thickness * 1e3:.0f} mm）")
        warn.set_text("   ".join(msgs))
        warn.set_visible(bool(msgs))

        x = s_grid[k] * 1e3
        cursor.set_data([x, x], [2.2, 6.6])

        ax3d.view_init(elev=26.0, azim=-62.0 + orbit * math.sin(2 * math.pi * k / n_frames))
        ax3d.set_xlim(*xlim)
        ax3d.set_ylim(*ylim)
        ax3d.set_zlim(*zlim)
        if progress and k % 30 == 0:
            print(f"    frame {k}/{n_frames}", flush=True)
        return ()

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, blit=False)
    out_path = str(out_path)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=4200)
        anim.save(out_path, writer=writer, dpi=dpi,
                  savefig_kwargs={"facecolor": C.bg})
    except (FileNotFoundError, RuntimeError):
        out_path = out_path.rsplit(".", 1)[0] + ".gif"
        anim.save(out_path, writer=animation.PillowWriter(fps=min(fps, 14)), dpi=dpi,
                  savefig_kwargs={"facecolor": C.bg})
    plt.close(fig)
    return out_path, report
