"""Animated render of a weld: what the process looks like, and what the
cameras see while it happens.

What this is
------------
A **visualisation of the same reduced-order physics** that runs everywhere else
in this repository, not a new simulation and emphatically not CFD.  The
temperature field is Rosenthal's analytic moving point source (``heat_source``)
evaluated on a grid in the torch frame; the pool cross-section is the ROM's
half-ellipse; the smoke plume is a procedural texture whose density is the
simulator's own smoke state.  Every number on screen comes from the log.

The figure is labelled accordingly, because a render that looks like CFD and
isn't is a liability in a proposal.

Why it earns its place
----------------------
Two things are much easier to see than to tabulate:

* the **baseline and the adaptive controller welding the same seam side by
  side**, in space rather than in time, so the burn-through happens where the
  gap opens and the clocks show what the cycle time actually cost;
* the **camera panels** — the RGB view greying out and blooming exactly when
  there is something worth looking at, next to an IR view that survives.  That
  is the Phase 2 measurement, rendered.

Playback is indexed by **position along the seam**, not by time, so the two
arms stay synchronised even though they take different numbers of seconds to
get there.  Each arm carries its own elapsed-time clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from weldloop.config import WeldConfig
from weldloop.physics.heat_source import rosenthal_thick_plate
from weldloop.viz.style import C, use_style

__all__ = ["RenderData", "prepare", "render_animation"]

#: window around the torch in the top view [m]
WIN_BACK, WIN_AHEAD, WIN_HALF_Y = 0.060, 0.020, 0.016
NX, NY = 170, 66


# --------------------------------------------------------------------------
@dataclass(slots=True)
class RenderData:
    """One arm's trajectory resampled onto a common seam-position axis."""

    name: str
    s: np.ndarray
    t: np.ndarray
    gap: np.ndarray
    p: np.ndarray
    w: np.ndarray
    T_pool: np.ndarray
    fill: np.ndarray
    V: np.ndarray
    I: np.ndarray
    smoke: np.ndarray
    burn_through: np.ndarray
    weave: np.ndarray
    v_travel: np.ndarray
    I_cmd: np.ndarray
    p_hat: np.ndarray
    p_std: np.ndarray


def _resample(table, s_grid: np.ndarray, col: str) -> np.ndarray:
    s = table["rb_s"]
    return np.interp(s_grid, s, table[col])


def prepare(runs: dict, cfg: WeldConfig, n_frames: int) -> tuple[np.ndarray, dict]:
    """Resample every arm onto a shared seam-position grid."""
    lengths = [d["table"]["rb_s"][-1] for d in runs.values()]
    s_grid = np.linspace(0.0, min(lengths), n_frames)
    out = {}
    for name, d in runs.items():
        t = d["table"]
        ctl = d.get("controller")
        if getattr(ctl, "history", None):
            hs = np.array([r["s"] for r in ctl.history])
            p_hat = np.interp(s_grid, hs, [r["p_hat"] for r in ctl.history])
            p_std = np.interp(s_grid, hs, [r["p_std"] for r in ctl.history])
        else:
            p_hat = np.full(n_frames, np.nan)
            p_std = np.full(n_frames, np.nan)
        out[name] = RenderData(
            name=name,
            s=s_grid,
            t=_resample(t, s_grid, "t"),
            gap=_resample(t, s_grid, "truth_gap"),
            p=_resample(t, s_grid, "truth_penetration"),
            w=_resample(t, s_grid, "truth_pool_w"),
            T_pool=_resample(t, s_grid, "truth_T_pool"),
            fill=_resample(t, s_grid, "truth_fill"),
            V=_resample(t, s_grid, "ps_V"),
            I=_resample(t, s_grid, "ps_I"),
            smoke=_resample(t, s_grid, "truth_smoke"),
            burn_through=_resample(t, s_grid, "truth_burn_through"),
            weave=_resample(t, s_grid, "rb_weave_offset"),
            I_cmd=_resample(t, s_grid, "cmd_I_set"),
            v_travel=_resample(t, s_grid, "rb_v_travel"),
            p_hat=p_hat,
            p_std=p_std,
        )
    return s_grid, out


# --------------------------------------------------------------------------
class _Plume:
    """Weld fume as a handful of puffs shed from the arc and drifting back.

    A texture lookup was the first attempt and it read as blocky wallpaper.
    Discrete puffs are cheaper, look like fume, and have an honest structure:
    each one is born at the arc, drifts backwards, spreads and fades, so the
    plume is dense where the arc is and ragged behind it.
    """

    N_PUFFS = 7

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.phase0 = rng.random(self.N_PUFFS)
        self.y0 = rng.normal(0.0, 0.0018, self.N_PUFFS)
        self.scale = rng.uniform(0.75, 1.35, self.N_PUFFS)

    def density(self, XI: np.ndarray, Y: np.ndarray, smoke: float, phase: float):
        out = np.zeros_like(XI)
        for j in range(self.N_PUFFS):
            age = (phase + self.phase0[j]) % 1.0          # 0 = just shed
            xi_c = -0.003 - 0.055 * age                    # drifts behind the arc
            sx = 0.0038 + 0.016 * age
            sy = 0.0026 + 0.011 * age
            amp = self.scale[j] * (1.0 - age) ** 2.0
            out += amp * np.exp(
                -0.5 * ((XI - xi_c) / sx) ** 2
                - 0.5 * ((Y - self.y0[j]) / sy) ** 2
            )
        return np.clip(0.46 * smoke * out, 0.0, 0.72)


class _TopView:
    """Plate seen from above: thermal field, seam, bead, torch, smoke."""

    def __init__(self, ax, cfg: WeldConfig, colour: str, label: str, seed: int) -> None:
        self.ax = ax
        self.cfg = cfg
        self.colour = colour
        self.plume = _Plume(seed)
        xi = np.linspace(-WIN_BACK, WIN_AHEAD, NX)
        y = np.linspace(-WIN_HALF_Y, WIN_HALF_Y, NY)
        self.XI, self.Y = np.meshgrid(xi, y, indexing="ij")
        self.extent = (-WIN_BACK * 1e3, WIN_AHEAD * 1e3,
                       -WIN_HALF_Y * 1e3, WIN_HALF_Y * 1e3)

        self.im = ax.imshow(
            np.zeros((NY, NX)), extent=self.extent, origin="lower",
            cmap="inferno", vmin=300.0, vmax=2400.0, aspect="auto", zorder=1,
        )
        self.smoke_im = ax.imshow(
            np.zeros((NY, NX, 4)), extent=self.extent, origin="lower",
            aspect="auto", zorder=4,
        )
        (self.seam_a,) = ax.plot([], [], color=C.text, lw=1.0, alpha=0.55, zorder=2)
        (self.seam_b,) = ax.plot([], [], color=C.text, lw=1.0, alpha=0.55, zorder=2)
        (self.bead,) = ax.plot([], [], color="#ffd7a0", lw=0.0, zorder=3)
        self.bead_fill = None
        (self.torch,) = ax.plot(
            [], [], marker="v", color=colour, ms=11, mew=1.2,
            mec=C.bg, zorder=6, ls="none",
        )
        (self.arc,) = ax.plot(
            [], [], marker="o", color="#fff4d6", ms=7, ls="none", zorder=5, alpha=0.95
        )
        self.txt = ax.text(
            0.012, 0.93, "", transform=ax.transAxes, color=C.text,
            fontsize=9, va="top", zorder=7,
            bbox=dict(facecolor=C.bg, edgecolor=C.grid, alpha=0.75, pad=3),
        )
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        ax.set_ylabel("y [mm]")
        ax.set_title(label, loc="left", fontsize=10.5, color=colour)
        ax.grid(False)
        ax.set_xticks([])

    def update(self, d: RenderData, k: int, seam, phase: float) -> None:
        cfg = self.cfg
        Q = cfg.pool.eta_arc * d.V[k] * d.I[k]
        v = max(d.v_travel[k], 1e-4)
        T = rosenthal_thick_plate(self.XI, self.Y - d.weave[k], 0.0, Q, v, cfg.material)
        self.im.set_data(np.clip(T, 300.0, 2400.0).T)

        # seam edges as they actually run through the window
        s_abs = d.s[k] + np.linspace(-WIN_BACK, WIN_AHEAD, 120)
        g = np.asarray(seam.gap_at(np.clip(s_abs, 0.0, seam.length)))
        x_mm = np.linspace(-WIN_BACK, WIN_AHEAD, 120) * 1e3
        self.seam_a.set_data(x_mm, 0.5 * g * 1e3)
        self.seam_b.set_data(x_mm, -0.5 * g * 1e3)

        # solidified bead behind the torch, width = pool width at that station
        back = np.linspace(-WIN_BACK, 0.0, 80)
        s_back = np.clip(d.s[k] + back, 0.0, seam.length)
        w_back = np.interp(s_back, d.s, d.w)
        if self.bead_fill is not None:
            self.bead_fill.remove()
        self.bead_fill = self.ax.fill_between(
            back * 1e3, -0.5 * w_back * 1e3, 0.5 * w_back * 1e3,
            color="#ffcf9a", alpha=0.42, lw=0, zorder=3,
        )

        self.torch.set_data([0.0], [d.weave[k] * 1e3])
        self.arc.set_data([0.0], [d.weave[k] * 1e3])

        # smoke: puffs shed from the arc, density from the simulator's own state
        density = self.plume.density(self.XI, self.Y - d.weave[k], d.smoke[k], phase)
        rgba = np.empty((NX, NY, 4))
        rgba[..., 0] = 0.86
        rgba[..., 1] = 0.85
        rgba[..., 2] = 0.82
        rgba[..., 3] = density
        self.smoke_im.set_data(np.transpose(rgba, (1, 0, 2)))

        bt = d.burn_through[k] > 0.5
        self.txt.set_text(
            f"t = {d.t[k]:5.1f} s     v = {d.v_travel[k] * 1e3:4.2f} mm/s     "
            f"I = {d.I[k]:3.0f} A     熔深 = {d.p[k] * 1e3:4.2f} mm"
            + ("     [烧穿]" if bt else "")
        )
        self.txt.set_color(C.bad if bt else C.text)


class _CrossSection:
    """The joint cut across the seam: plate, gap, pool, crown, burn-through.

    Drawn as explicit polygons at true proportions (``aspect='equal'``), so
    what the audience sees is the actual bead geometry the ROM produced, not a
    stylised cartoon of it.
    """

    def __init__(self, ax, cfg: WeldConfig, colour: str) -> None:
        self.ax = ax
        self.cfg = cfg
        self.colour = colour
        h = cfg.joint.thickness * 1e3
        ax.set_xlim(-8.0, 8.0)
        ax.set_ylim(h + 2.5, -3.5)          # depth positive downwards
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel("depth [mm]", fontsize=8.5)
        ax.set_xlabel("y [mm]", fontsize=8)
        ax.set_title("横截面", fontsize=9.5, loc="left")
        ax.tick_params(labelsize=7.5)
        ax.grid(False)
        self.artists: list = []
        self.txt = ax.text(
            0.03, 0.02, "", transform=ax.transAxes, color=C.text, fontsize=8
        )

    @staticmethod
    def _half_ellipse(a: float, b: float, down: bool, n: int = 60):
        """Polygon of a half ellipse of semi-width ``a`` and depth ``b``."""
        th = np.linspace(0.0, math.pi, n)
        x = a * np.cos(th)
        z = b * np.sin(th)
        return np.column_stack([x, z if down else -z])

    def update(self, d: RenderData, k: int) -> None:
        import matplotlib.patches as mp

        for art in self.artists:
            art.remove()
        self.artists.clear()
        ax = self.ax
        h = self.cfg.joint.thickness * 1e3
        gap = max(d.gap[k] * 1e3, 0.0)
        p = d.p[k] * 1e3
        w = d.w[k] * 1e3
        fill = d.fill[k]
        burnt = d.burn_through[k] > 0.5

        # parent plate either side of the root gap
        for x0, x1 in ((-8.0, -gap / 2), (gap / 2, 8.0)):
            if x1 > x0:
                self.artists.append(
                    ax.add_patch(
                        mp.Rectangle((x0, 0.0), x1 - x0, h,
                                     facecolor="#454c56", edgecolor=C.grid, lw=0.8,
                                     zorder=1)
                    )
                )
        # deposited crown above the surface, and the filler plugging the gap
        crown_h = 0.6 + 1.5 * min(fill, 1.8)
        crown = self._half_ellipse(max(w / 2.0, gap / 2.0 + 1.2), crown_h, down=False)
        self.artists.append(
            ax.add_patch(mp.Polygon(crown, closed=True, facecolor="#c98a4b",
                                    edgecolor="#e8b782", lw=0.9, zorder=2))
        )
        if gap > 0.0:
            plug_depth = min(fill, 1.0) * h
            self.artists.append(
                ax.add_patch(
                    mp.Rectangle((-gap / 2, 0.0), gap, plug_depth,
                                 facecolor="#c98a4b", edgecolor="none", zorder=2)
                )
            )
        # the molten pool itself
        pool = self._half_ellipse(w / 2.0, p, down=True)
        self.artists.append(
            ax.add_patch(mp.Polygon(pool, closed=True, facecolor="#ff8f2e",
                                    edgecolor="#ffe0b0", lw=1.1, zorder=3))
        )
        # hotter core, for readability rather than physics
        core = self._half_ellipse(0.62 * w / 2.0, 0.62 * p, down=True)
        self.artists.append(
            ax.add_patch(mp.Polygon(core, closed=True, facecolor="#ffd08a",
                                    edgecolor="none", alpha=0.85, zorder=4))
        )
        # plate underside, and the burn-through breach
        self.artists.append(
            ax.add_line(
                __import__("matplotlib").lines.Line2D(
                    [-8.0, 8.0], [h, h], color=C.muted, lw=1.0, zorder=5
                )
            )
        )
        if burnt:
            self.artists.append(
                ax.add_patch(
                    mp.Rectangle((-w / 2, h - 0.5), w, 1.9, facecolor=C.bad,
                                 alpha=0.55, lw=0, zorder=6)
                )
            )
            self.artists.append(
                ax.text(0.0, h + 2.0, "烧穿", color=C.bad, fontsize=10,
                        ha="center", va="center", zorder=7)
            )
        self.txt.set_text(
            f"间隙 {gap:.2f}  熔宽 {w:.1f}\n熔深 {p:.2f}/{h:.0f}  填充 {fill:.2f}"
        )

def _camera_frames(d: RenderData, k: int, cfg: WeldConfig, rng) -> tuple[np.ndarray, np.ndarray]:
    """Procedural RGB and IR frames of the pool, degraded per Phase 2."""
    nx, ny = 116, 84
    x = np.linspace(-14.0, 14.0, nx)
    y = np.linspace(-10.0, 10.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    w, p = d.w[k] * 1e3, d.p[k] * 1e3

    pool = np.exp(-((X / (0.5 * w)) ** 2 + (Y / (0.34 * w)) ** 2) ** 1.4)
    arc = np.exp(-((X / 1.5) ** 2 + (Y / 1.5) ** 2))

    # --- RGB: smoke transmission x arc glare, both from the measured model
    vis = math.exp(-cfg.sensors.rgb_smoke_tau * d.smoke[k])
    glare = 1.0 / (1.0 + (d.I[k] / cfg.sensors.rgb_glare_I) ** 2)
    quality = vis * glare
    img = np.zeros((nx, ny, 3))
    img[..., 0] = 0.85 * pool + 1.6 * arc
    img[..., 1] = 0.45 * pool + 1.5 * arc
    img[..., 2] = 0.12 * pool + 1.4 * arc
    img *= vis
    img += (1.0 - vis) * 0.62                      # fume veil
    img += 0.55 * (1.0 - glare) * arc[..., None]   # bloom
    img += 0.035 * rng.standard_normal((nx, ny, 1))
    rgb = np.clip(np.transpose(img, (1, 0, 2)), 0.0, 1.0)

    # --- IR: same scene, far weaker attenuation, mapped through a colormap
    atten = math.exp(
        -cfg.sensors.ir_smoke_tau * (d.smoke[k] - cfg.sensors.smoke_ref)
    ) ** cfg.sensors.ir_width_atten_exp
    T = 300.0 + (d.T_pool[k] - 300.0) * pool * atten
    T += 25.0 * rng.standard_normal((nx, ny))
    return rgb, T.T, quality


# --------------------------------------------------------------------------
def render_animation(
    runs: dict,
    cfg: WeldConfig,
    seam,
    out_path,
    *,
    n_frames: int = 420,
    fps: int = 21,
    dpi: int = 108,
    progress: bool = True,
):
    """Render the side-by-side animation to ``out_path`` (mp4, or gif fallback)."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    use_style()
    s_grid, data = prepare(runs, cfg, n_frames)
    names = [n for n in ("baseline", "adaptive") if n in data]
    colours = {"baseline": C.baseline, "adaptive": C.adaptive}
    labels = {
        "baseline": "定参数 baseline —— 固定电流/速度",
        "adaptive": "自适应 adaptive —— 电源在环",
    }

    fig = plt.figure(figsize=(14.6, 8.2))
    gs = fig.add_gridspec(
        3, 3, width_ratios=[1.95, 0.80, 1.05], height_ratios=[1.16, 1.16, 0.78],
        hspace=0.34, wspace=0.20,
        left=0.042, right=0.975, top=0.885, bottom=0.075,
    )
    tops, sections = {}, {}
    for i, name in enumerate(names):
        tops[name] = _TopView(
            fig.add_subplot(gs[i, 0]), cfg, colours[name], labels[name], seed=17 + i
        )
        sections[name] = _CrossSection(fig.add_subplot(gs[i, 1]), cfg, colours[name])

    ax_rgb = fig.add_subplot(gs[0, 2])
    ax_ir = fig.add_subplot(gs[1, 2])
    ax_pen = fig.add_subplot(gs[2, 0:2])
    ax_vi = fig.add_subplot(gs[2, 2])

    rng = np.random.default_rng(4)
    d_ad = data.get("adaptive", data[names[0]])
    rgb0, ir0, q0 = _camera_frames(d_ad, 0, cfg, rng)
    im_rgb = ax_rgb.imshow(rgb0, origin="lower", aspect="auto")
    im_ir = ax_ir.imshow(ir0, origin="lower", aspect="auto", cmap="inferno",
                         vmin=300.0, vmax=2400.0)
    ir_contour = {"art": None}
    ir_txt = ax_ir.text(
        0.03, 0.05, "", transform=ax_ir.transAxes, color="#ffffff", fontsize=9,
        bbox=dict(facecolor="#000000", alpha=0.45, pad=2, edgecolor="none"),
    )
    for ax, title in ((ax_rgb, "RGB 相机"), (ax_ir, "IR 热像仪")):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(title, fontsize=10, loc="left")
    txt_rgb = ax_rgb.text(
        0.03, 0.05, "", transform=ax_rgb.transAxes, color="#ffffff", fontsize=9,
        bbox=dict(facecolor="#000000", alpha=0.45, pad=2, edgecolor="none"),
    )

    # penetration strip: both arms against seam position
    for name in names:
        ax_pen.plot(data[name].s * 1e3, data[name].p * 1e3, color=colours[name], lw=1.2,
                    label=labels[name].split(" ")[0])
    ax_pen.axhspan(cfg.control.p_lo * 1e3, cfg.control.p_hi * 1e3,
                   color=C.band_ok, alpha=0.13, lw=0)
    ax_pen.axhline(cfg.joint.thickness * 1e3, color=C.bad, ls="--", lw=1.0)
    ax_pen.set_ylim(2.2, 6.6)
    ax_pen.set_ylabel("真实熔深 [mm]", fontsize=8.5)
    ax_pen.set_xlabel("沿焊缝位置 s [mm]", fontsize=8.5)
    ax_pen.legend(loc="lower right", fontsize=7)
    ax_pen.tick_params(labelsize=7.5)
    (cur_pen,) = ax_pen.plot([], [], color=C.text, lw=1.0, alpha=0.8)

    ax_vi.plot(d_ad.s * 1e3, d_ad.I_cmd, color=C.warn, lw=1.1, label="I [A]")
    ax_vi.set_ylabel("自适应电流指令 [A]", fontsize=8.5)
    ax_vi.set_xlabel("沿焊缝位置 s [mm]", fontsize=8.5)
    ax_vi.tick_params(labelsize=7.5)
    ax_vi2 = ax_vi.twinx()
    ax_vi2.plot(d_ad.s * 1e3, d_ad.gap * 1e3, color=C.aux2, lw=1.0, ls="--")
    ax_vi2.set_ylabel("间隙 [mm]", fontsize=8.5, color=C.aux2)
    ax_vi2.tick_params(labelsize=7.5, colors=C.aux2)
    ax_vi2.grid(False)
    (cur_vi,) = ax_vi.plot([], [], color=C.text, lw=1.0, alpha=0.8)

    fig.suptitle(
        "weldloop —— 同一条变间隙焊缝，两种控制方式（按位置同步播放）\n"
        "温度场为 Rosenthal 解析解、熔池为降阶模型半椭圆、烟羽为程序化烟团；"
        "这是同一套降阶物理的可视化，不是 CFD",
        color=C.text, fontsize=11.5, fontweight="bold", y=0.985,
    )

    def frame(k: int):
        phase = k / max(n_frames - 1, 1)
        for name in names:
            tops[name].update(data[name], k, seam, (phase * 26.0) % 1.0)
            sections[name].update(data[name], k)
        rgb, ir, q = _camera_frames(d_ad, k, cfg, rng)
        im_rgb.set_data(rgb)
        im_ir.set_data(ir)
        if ir_contour["art"] is not None:
            ir_contour["art"].remove()
        ir_contour["art"] = ax_ir.contour(
            ir, levels=[cfg.material.T_m], colors=["#7ef7d0"], linewidths=1.2
        )
        ir_txt.set_text(f"熔合等温线可见   峰值 {ir.max():4.0f} K")
        usable = q >= cfg.sensors.rgb_quality_min
        txt_rgb.set_text(f"可用度 {q:.3f}" + ("" if usable else "   不可用"))
        txt_rgb.set_color("#ffffff" if usable else "#ff7b7b")
        x = s_grid[k] * 1e3
        cur_pen.set_data([x, x], [2.2, 6.6])
        cur_vi.set_data([x, x], [d_ad.I_cmd.min(), d_ad.I_cmd.max()])
        if progress and k % 40 == 0:
            print(f"    frame {k}/{n_frames}", flush=True)
        return ()

    anim = animation.FuncAnimation(fig, frame, frames=n_frames, blit=False)
    out_path = str(out_path)
    try:
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=3600,
            metadata={"title": "weldloop", "comment": "reduced-order physics visualisation"},
        )
        anim.save(out_path, writer=writer, dpi=dpi)
    except (FileNotFoundError, RuntimeError):
        out_path = out_path.rsplit(".", 1)[0] + ".gif"
        anim.save(out_path, writer=animation.PillowWriter(fps=min(fps, 15)), dpi=dpi)
    plt.close(fig)
    return out_path
