"""The fit-up inspection report — the *document* the planning layer reads.

Deliberately not styled like the rest of the demo.  Every other figure here is
an output we drew for ourselves; this one is an input that arrives from
somebody else, so it is drawn the way a real pre-weld inspection sheet looks:
white paper, a header block, a scanned gap trace with a millimetre grid, and a
section view off the drawing.

That matters for the planning demo.  The task layer is given this image and
the job card text and nothing else — no arrays.  If the plan it produces cuts
the seam in the right places, it did so by reading a plot, which is the actual
claim being tested.

TODO(real-hw): replace with the real scanner's PDF/PNG export and the real
job card.  Nothing downstream parses this image, so its layout is free.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fitup_report_figure"]

_INK = "#1b1e23"
_PAPER = "#f6f4ef"
_RULE = "#c9c4b8"
_TRACE = "#1f5f8b"
_STEEL = "#b9bcc1"
_STEEL_DARK = "#9aa0a6"


def fitup_report_figure(seam, cfg, *, job_id: str = "WL-2409-017", scan_date: str = ""):
    """Render the fit-up report for ``seam``.

    Parameters
    ----------
    seam:
        The measured joint.  Its gap, misalignment and thickness arrays are
        what the scanner and the drawing between them would report.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from weldloop.viz.style import cjk_font

    s_mm = seam.s * 1e3
    gap_mm = seam.gap * 1e3
    off_mm = seam.offset * 1e3
    L = float(s_mm[-1])
    h_mm = (
        seam.thickness * 1e3
        if seam.thickness is not None
        else np.full_like(s_mm, cfg.joint.thickness * 1e3)
    )

    fig = _draw(
        fig_spec=(s_mm, gap_mm, off_mm, h_mm, L), seam=seam, cfg=cfg,
        job_id=job_id, scan_date=scan_date,
    )
    _embed_cjk(fig)
    return fig


def _embed_cjk(fig) -> None:
    """Pin a CJK face onto every text artist in the figure.

    Setting rcParams would be simpler but does not survive: matplotlib
    resolves fonts at *draw* time, and this figure is saved by the caller,
    long after any rc_context has closed.  Fallback order is kept per artist,
    so the monospace header block stays column-aligned in its ASCII half.
    """
    import matplotlib as mpl
    from matplotlib.text import Text

    from weldloop.viz.style import cjk_font

    cjk = cjk_font()
    if cjk is None:  # pragma: no cover - depends on installed fonts
        return
    for artist in fig.findobj(Text):
        generic = list(artist.get_fontfamily())
        concrete: list[str] = []
        for fam in generic:
            key = f"font.{fam}"
            concrete += mpl.rcParams[key] if key in mpl.rcParams else [fam]
        artist.set_fontfamily(concrete + [cjk])


def _draw(*, fig_spec, seam, cfg, job_id, scan_date):
    import matplotlib.pyplot as plt

    s_mm, gap_mm, off_mm, h_mm, L = fig_spec

    fig = plt.figure(figsize=(11.0, 7.4), facecolor=_PAPER)
    gs = fig.add_gridspec(
        4, 1, height_ratios=[0.55, 1.35, 1.9, 1.0],
        left=0.075, right=0.965, top=0.955, bottom=0.075, hspace=0.55,
    )

    # -- header block -----------------------------------------------------
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.text(0.0, 0.85, "坡口装配检测报告  /  JOINT FIT-UP INSPECTION",
            fontsize=15, fontweight="bold", color=_INK, va="top")
    left = (
        f"工件号 JOB      {job_id}\n"
        f"接头 JOINT      对接 square butt, 单道 single pass\n"
        f"材料 MATERIAL   低碳钢 mild steel"
    )
    right = (
        f"焊缝长度 LENGTH  {L:.0f} mm\n"
        f"扫描设备 DEVICE  laser profiler, 0.5 mm pitch\n"
        f"扫描日期 DATE    {scan_date or '—'}"
    )
    ax.text(0.0, 0.34, left, fontsize=9.5, color=_INK, va="top", family="monospace")
    ax.text(0.52, 0.34, right, fontsize=9.5, color=_INK, va="top", family="monospace")
    ax.axhline(0.0, color=_RULE, lw=1.2)

    # -- section view: what the drawing says the plate is ------------------
    ax = fig.add_subplot(gs[1], facecolor=_PAPER)
    top = np.zeros_like(s_mm)
    ax.fill_between(s_mm, top, -h_mm, color=_STEEL, lw=0.0, step=None)
    ax.plot(s_mm, -h_mm, color=_INK, lw=1.3)
    ax.plot(s_mm, top, color=_INK, lw=1.3)
    # dimension the two thicknesses that actually occur
    for value in np.unique(np.round(h_mm, 3)):
        m = np.isclose(h_mm, value, atol=1e-6)
        mid = float(np.mean(s_mm[m]))
        ax.annotate("", xy=(mid, 0.0), xytext=(mid, -value),
                    arrowprops=dict(arrowstyle="<->", color=_INK, lw=1.0))
        ax.text(mid + 0.012 * L, -0.5 * value, f"{value:.0f} mm",
                fontsize=10, color=_INK, va="center", fontweight="bold")
    if len(np.unique(np.round(h_mm, 3))) > 1:
        i_step = int(np.argmax(np.abs(np.diff(h_mm)))) + 1
        ax.axvline(s_mm[i_step], color=_INK, lw=1.0, ls=(0, (4, 3)))
        ax.text(s_mm[i_step], 0.6, f"板厚台阶 step @ {s_mm[i_step]:.0f} mm",
                fontsize=9, color=_INK, ha="center", va="bottom")
    ax.set_ylim(-1.35 * float(h_mm.max()), 1.9)
    ax.set_xlim(0, L)
    ax.set_title("剖视图 SECTION A-A（按图纸 per drawing）", loc="left",
                 fontsize=10.5, color=_INK, fontweight="bold")
    ax.set_yticks([])
    _paper_axes(ax)

    # -- the scan ---------------------------------------------------------
    ax = fig.add_subplot(gs[2], facecolor="#ffffff")
    ax.plot(s_mm, gap_mm, color=_TRACE, lw=1.7)
    ax.fill_between(s_mm, 0.0, gap_mm, color=_TRACE, alpha=0.13, lw=0)
    ax.set_xlim(0, L)
    ax.set_ylim(0, max(1.0, float(gap_mm.max()) * 1.25))
    ax.set_ylabel("根部间隙 root gap  [mm]", fontsize=9.5, color=_INK)
    ax.set_title("激光扫描 MEASURED ROOT GAP g(s)", loc="left",
                 fontsize=10.5, color=_INK, fontweight="bold")
    ax.grid(True, which="major", color=_RULE, lw=0.7, alpha=0.9)
    ax.grid(True, which="minor", color=_RULE, lw=0.4, alpha=0.5)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(plt.MultipleLocator(20.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    _paper_axes(ax)

    # -- misalignment -----------------------------------------------------
    ax = fig.add_subplot(gs[3], facecolor="#ffffff")
    ax.plot(s_mm, off_mm, color="#7a6a9b", lw=1.4)
    ax.axhline(0.0, color=_RULE, lw=0.9)
    ax.set_xlim(0, L)
    ax.set_xlabel("沿焊缝位置 position along seam  s  [mm]", fontsize=9.5, color=_INK)
    ax.set_ylabel("错边 misalign\n[mm]", fontsize=9, color=_INK)
    ax.grid(True, color=_RULE, lw=0.7, alpha=0.9)
    ax.xaxis.set_major_locator(plt.MultipleLocator(20.0))
    _paper_axes(ax)

    return fig


def _paper_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color(_RULE)
    ax.tick_params(colors=_INK, labelsize=8.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(_INK)
