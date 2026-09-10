"""Phase 5/6 tests: the figures and the animation actually build.

These are smoke tests with teeth: they assert that every figure renders from
real simulation output, that the demo script writes the files it promises,
and that the renderer's data preparation lines the two controllers up in
*space* rather than in time — which is the whole reason the side-by-side
comparison is fair.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from weldloop.config import default_config
from weldloop.control.baseline import BaselineController
from weldloop.control.motion_layer import AdaptiveController
from weldloop.metrics import score
from weldloop.sim.runner import simulate
from weldloop.viz import render as render_mod
from weldloop.viz.dashboard import _bin_stat, dashboard_figure, summary_figure
from weldloop.viz.style import C, cjk_font, downsample, use_style

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def two_arms():
    cfg = default_config(seam__kind="step", seam__length=0.06, sim__seed=0)
    runs = {}
    for name, ctl in (
        ("baseline", BaselineController(cfg)),
        ("adaptive", AdaptiveController(cfg)),
    ):
        res = simulate(cfg, seed=0, controller=ctl)
        runs[name] = {
            "table": res.table, "controller": ctl,
            "metrics": score(res.table, cfg, name),
        }
    return cfg, runs, res.seam


# ==========================================================================
class TestStyle:
    def test_palette_reserves_orange_for_the_adaptive_controller(self):
        assert C.adaptive == C.estimate  # the accent, used for both
        assert C.baseline != C.adaptive
        assert C.bad != C.adaptive

    def test_downsample_thins_without_reordering(self):
        x = np.arange(10_000, dtype=float)
        y = x * 2.0
        xs, ys = downsample(x, y, n=500)
        assert len(xs) <= 500
        assert np.all(np.diff(xs) > 0)
        assert np.allclose(ys, xs * 2.0)

    def test_downsample_leaves_short_arrays_alone(self):
        x = np.arange(10, dtype=float)
        assert np.array_equal(downsample(x, n=500), x)

    def test_cjk_font_lookup_never_raises(self):
        name = cjk_font()
        assert name is None or isinstance(name, str)

    def test_use_style_sets_a_charcoal_background(self):
        use_style()
        assert matplotlib.rcParams["figure.facecolor"] == C.bg


class TestBinStat:
    def test_bins_preserve_the_mean(self):
        x = np.linspace(0.0, 10.0, 5000)
        y = np.sin(x)
        cx, cy = _bin_stat(x, y, 50)
        assert len(cx) == len(cy) == 50
        assert abs(np.nanmean(cy) - np.mean(y)) < 0.02

    def test_max_preserves_the_excursion_a_mean_would_hide(self):
        x = np.linspace(0.0, 1.0, 1000)
        y = np.zeros(1000)
        y[500] = 7.0
        _, mx = _bin_stat(x, y, 20, "max")
        assert np.nanmax(mx) == pytest.approx(7.0)


# ==========================================================================
class TestFigures:
    def test_summary_figure_builds(self, two_arms, tmp_path):
        cfg, runs, _ = two_arms
        fig = summary_figure(runs, cfg, seed=0, gap_profile="step")
        assert len(fig.axes) >= 3
        out = tmp_path / "summary.png"
        fig.savefig(out, dpi=70)
        assert out.stat().st_size > 20_000

    def test_dashboard_figure_builds_for_both_arms(self, two_arms, tmp_path):
        cfg, runs, _ = two_arms
        fig = dashboard_figure(runs, cfg)
        out = tmp_path / "dash.png"
        fig.savefig(out, dpi=70)
        assert out.stat().st_size > 20_000

    def test_dashboard_tolerates_a_controller_with_no_estimator(self, two_arms):
        """The baseline has no history; the figure must say so, not crash."""
        cfg, runs, _ = two_arms
        assert not getattr(runs["baseline"]["controller"], "history", [])
        fig = dashboard_figure({"baseline": runs["baseline"]}, cfg)
        assert fig is not None


# ==========================================================================
class TestRenderPreparation:
    def test_arms_are_aligned_in_space_not_time(self, two_arms):
        """The point of the side-by-side: same position, different clocks."""
        cfg, runs, _ = two_arms
        s_grid, data = render_mod.prepare(runs, cfg, n_frames=60)
        assert np.all(np.diff(s_grid) > 0)
        for d in data.values():
            assert np.array_equal(d.s, s_grid)
            assert np.all(np.diff(d.t) >= -1e-9)      # each clock runs forward
        # ...and at the same station the two clocks generally differ
        assert not np.allclose(data["baseline"].t, data["adaptive"].t)

    def test_every_channel_is_finite(self, two_arms):
        cfg, runs, _ = two_arms
        _, data = render_mod.prepare(runs, cfg, n_frames=40)
        d = data["adaptive"]
        for name in ("gap", "p", "w", "T_pool", "V", "I", "smoke", "I_cmd"):
            assert np.all(np.isfinite(getattr(d, name))), name

    def test_a_controller_without_an_estimator_yields_nan_beliefs(self, two_arms):
        cfg, runs, _ = two_arms
        _, data = render_mod.prepare(runs, cfg, n_frames=40)
        assert np.all(np.isnan(data["baseline"].p_hat))

    def test_plume_density_is_bounded_and_tracks_smoke(self):
        plume = render_mod._Plume(seed=3)
        XI, Y = np.meshgrid(
            np.linspace(-0.06, 0.02, 40), np.linspace(-0.016, 0.016, 20), indexing="ij"
        )
        thin = plume.density(XI, Y, 0.2, 0.3)
        thick = plume.density(XI, Y, 1.2, 0.3)
        assert 0.0 <= thin.min() and thick.max() <= 0.80
        assert thick.sum() > thin.sum()

    def test_camera_frames_show_ir_beating_rgb(self, two_arms):
        """The Phase 2 result, as rendered pixels."""
        cfg, runs, _ = two_arms
        _, data = render_mod.prepare(runs, cfg, n_frames=20)
        rng = np.random.default_rng(0)
        rgb, ir, quality = render_mod._camera_frames(data["adaptive"], 10, cfg, rng)
        assert rgb.shape[2] == 3 and rgb.min() >= 0.0 and rgb.max() <= 1.0
        assert quality < 0.4                       # the camera is not doing well
        assert ir.max() > cfg.material.T_m         # the IR frame still resolves the pool


class TestRenderAnimation:
    @pytest.mark.slow
    def test_a_short_animation_writes_a_playable_file(self, two_arms, tmp_path):
        cfg, runs, seam = two_arms
        out = render_mod.render_animation(
            runs, cfg, seam, tmp_path / "clip.mp4",
            n_frames=6, fps=6, dpi=52, progress=False,
        )
        path = Path(out)
        assert path.exists() and path.stat().st_size > 5_000
        assert path.suffix in (".mp4", ".gif")


# ==========================================================================
class TestRunDemoScript:
    @pytest.mark.slow
    def test_it_produces_everything_it_promises(self, tmp_path):
        out = tmp_path / "out"
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "run_demo.py"),
             "--seed", "0", "--gap-profile", "step", "--length", "0.05",
             "--out", str(out), "--no-residual"],
            capture_output=True, text=True, timeout=600, cwd=REPO,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        for f in ("summary.png", "dashboard.png", "metrics.json"):
            assert (out / f).exists(), f
        metrics = json.loads((out / "metrics.json").read_text())
        assert metrics["seed"] == 0
        assert set(metrics["controllers"]) == {"baseline", "adaptive"}
        assert metrics["improvement"]["penetration_std_ratio"] > 0.0
        assert metrics["residual_used"] is False
