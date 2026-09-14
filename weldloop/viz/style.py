"""Shared matplotlib style for every figure in the demo.

Charcoal panels, one orange accent reserved for the adaptive controller.  The
figures are meant to be dropped straight into a slide deck, so they are sized
for 16:9 and use font sizes that survive a projector.
"""

from __future__ import annotations

import matplotlib as mpl

__all__ = ["C", "cjk_font", "use_style", "downsample", "band", "shade_flag"]

#: Preferred CJK faces.  The README and every figure title are in Chinese
#: because the audience is a Chinese company, so a figure rendered with
#: DejaVu Sans alone comes out full of tofu boxes.  matplotlib falls back
#: through this list glyph by glyph.
_CJK_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Droid Sans Fallback",
    "Microsoft YaHei",
    "PingFang SC",
    "SimHei",
)


def cjk_font() -> str | None:
    """First CJK face actually installed, or ``None``.

    TODO(real-hw): on a machine with no CJK font the figures still render, but
    the Chinese labels become boxes.  Install ``fonts-noto-cjk``.
    """
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


class C:
    """The palette.  ``adaptive`` is the only orange in any figure."""

    bg = "#1b1e23"
    panel = "#23272e"
    grid = "#343a42"
    text = "#e6e8ea"
    muted = "#8c959f"

    adaptive = "#ff8c42"  # the accent: reserved for the adaptive controller
    baseline = "#5aa9e6"
    truth = "#e6e8ea"
    estimate = "#ff8c42"
    band_ok = "#5ecf8f"
    bad = "#e5484d"
    warn = "#f2c14e"
    aux = "#b48ead"
    aux2 = "#6fd3d0"


def use_style() -> None:
    """Apply the demo's rcParams globally."""
    families = ["DejaVu Sans"]
    cjk = cjk_font()
    if cjk is not None:
        families.insert(0, cjk)
    # CJK also has to be reachable from the monospace stack: several figures
    # set fixed-width text for column-aligned ledgers, and without a fallback
    # every Chinese glyph in one of those comes out as a box.
    monospace = ["DejaVu Sans Mono"] + ([cjk] if cjk is not None else [])
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": families,
            "font.monospace": monospace,
            "axes.unicode_minus": False,
            "figure.facecolor": C.bg,
            "savefig.facecolor": C.bg,
            "axes.facecolor": C.panel,
            "axes.edgecolor": C.grid,
            "axes.labelcolor": C.text,
            "axes.titlecolor": C.text,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": C.grid,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "text.color": C.text,
            "xtick.color": C.muted,
            "ytick.color": C.muted,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
            "figure.dpi": 130,
            "font.size": 9,
        }
    )


def downsample(*arrays, n: int = 4000):
    """Uniformly thin long traces so plotting stays fast and files stay small."""
    length = len(arrays[0])
    if length <= n:
        return arrays if len(arrays) > 1 else arrays[0]
    step = max(length // n, 1)
    out = tuple(a[::step] for a in arrays)
    return out if len(out) > 1 else out[0]


def band(ax, lo: float, hi: float, color: str = C.band_ok, label: str | None = None):
    """Shade an acceptance band across the full x range."""
    ax.axhspan(lo, hi, color=color, alpha=0.13, lw=0, label=label, zorder=0)


def shade_flag(ax, x, flag, color: str = C.bad, alpha: float = 0.28, label=None):
    """Shade the x-intervals where a boolean flag is set."""
    import numpy as np

    flag = np.asarray(flag, dtype=bool)
    if not flag.any():
        return
    edges = np.flatnonzero(np.diff(flag.astype(np.int8)))
    starts = list(edges[flag[edges + 1]] + 1)
    ends = list(edges[~flag[edges + 1]] + 1)
    if flag[0]:
        starts = [0] + starts
    if flag[-1]:
        ends = ends + [len(flag) - 1]
    first = True
    for a, b in zip(starts, ends):
        ax.axvspan(
            x[a], x[b], color=color, alpha=alpha, lw=0, zorder=0,
            label=label if first else None,
        )
        first = False
