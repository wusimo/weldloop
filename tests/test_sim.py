"""Phase 1 tests: seam generator and the ``WeldCell`` plant.

The important properties here are *determinism* (a demo that cannot be
reproduced is not evidence of anything) and that the cell reproduces the two
behaviours the proposal rests on: GMAW self-regulation, and a pool oscillation
that is actually visible in the 5 kHz voltage stream.
"""

from __future__ import annotations

import numpy as np
import pytest

from weldloop.config import default_config
from weldloop.physics.arc import melting_rate
from weldloop.sim.cell import OrnsteinUhlenbeck, TorchCommand, WeldCell
from weldloop.sim.seam import make_seam


def _short_cfg(**over):
    """A 60 mm seam — long enough to be a weld, short enough for a test."""
    over.setdefault("seam__length", 0.06)
    return default_config(**over)


def _collect(cell, keys):
    rows = []
    for gt in cell.run():
        rows.append([k(gt) for k in keys])
    return np.asarray(rows, dtype=float)


# ==========================================================================
# seam
# ==========================================================================
class TestSeam:
    @pytest.mark.parametrize("kind", ["constant", "step", "ramp", "sine", "random"])
    def test_profiles_are_deterministic_and_in_range(self, kind):
        cfg = default_config(seam__kind=kind)
        a = make_seam(cfg.seam, seed=5)
        b = make_seam(cfg.seam, seed=5)
        assert np.array_equal(a.gap, b.gap)
        assert np.array_equal(a.offset, b.offset)
        assert a.gap.min() >= 0.0
        assert a.gap.max() <= cfg.seam.gap_max + 6.0 * cfg.seam.noise_std
        assert np.abs(a.offset).max() == pytest.approx(cfg.seam.misalign_max)

    def test_different_seeds_give_different_seams(self):
        cfg = default_config(seam__kind="random")
        assert not np.array_equal(make_seam(cfg.seam, 1).gap, make_seam(cfg.seam, 2).gap)

    def test_step_profile_actually_spans_the_gap_band(self):
        cfg = default_config(seam__kind="step")
        seam = make_seam(cfg.seam, 0)
        assert seam.gap.max() - seam.gap.min() > 0.6 * (cfg.seam.gap_max - cfg.seam.gap_min)

    def test_gap_at_interpolates_and_clamps(self):
        cfg = default_config(seam__kind="ramp")
        seam = make_seam(cfg.seam, 0)
        assert seam.gap_at(-1.0) == pytest.approx(seam.gap[0])
        assert seam.gap_at(seam.length + 1.0) == pytest.approx(seam.gap[-1])
        mid = 0.5 * (seam.s[10] + seam.s[11])
        assert seam.gap_at(mid) == pytest.approx(0.5 * (seam.gap[10] + seam.gap[11]))

    def test_smoothing_keeps_the_profile_pool_scale_smooth(self):
        """Consecutive stations must not jump by more than the fit-up noise."""
        cfg = default_config(seam__kind="step")
        seam = make_seam(cfg.seam, 0)
        jumps = np.abs(np.diff(seam.gap))
        assert np.median(jumps) < 3.0 * cfg.seam.noise_std


class TestOrnsteinUhlenbeck:
    def test_is_zero_mean_with_the_requested_std(self):
        ou = OrnsteinUhlenbeck(0.5, 1.0e-3, np.random.default_rng(0))
        xs = [ou.step(1.0e-3) for _ in range(400_000)]
        assert abs(np.mean(xs)) < 1.0e-4
        assert np.std(xs) == pytest.approx(1.0e-3, rel=0.15)

    def test_is_deterministic(self):
        def run(seed):
            ou = OrnsteinUhlenbeck(0.5, 1.0e-3, np.random.default_rng(seed))
            return [ou.step(1.0e-3) for _ in range(100)]

        assert run(3) == run(3)


# ==========================================================================
# cell
# ==========================================================================
class TestWeldCellDeterminism:
    def test_same_seed_gives_an_identical_trajectory(self):
        keys = [lambda g: g.pool.p, lambda g: g.arc.V, lambda g: g.arc.I]
        a = _collect(WeldCell(_short_cfg(), seed=0), keys)
        b = _collect(WeldCell(_short_cfg(), seed=0), keys)
        assert a.shape == b.shape
        assert np.array_equal(a, b)

    def test_different_seeds_diverge(self):
        keys = [lambda g: g.arc.V]
        a = _collect(WeldCell(_short_cfg(), seed=0), keys)
        b = _collect(WeldCell(_short_cfg(), seed=1), keys)
        assert not np.array_equal(a[: min(len(a), len(b))], b[: min(len(a), len(b))])

    def test_reset_restores_the_initial_trajectory(self):
        cell = WeldCell(_short_cfg(), seed=0)
        first = [cell.step().arc.V for _ in range(500)]
        cell.reset()
        again = [cell.step().arc.V for _ in range(500)]
        assert first == again


class TestWeldCellKinematics:
    def test_run_terminates_at_the_end_of_the_seam(self):
        cfg = _short_cfg()
        cell = WeldCell(cfg, seed=0)
        gts = list(cell.run())
        assert cell.done
        assert gts[-1].s == pytest.approx(cell.seam.length, abs=1e-3)
        expected = cell.seam.length / cfg.baseline.v_travel
        assert 0.7 * expected < gts[-1].t < 1.4 * expected

    def test_position_is_monotone_and_the_clock_is_uniform(self):
        cfg = _short_cfg()
        r = _collect(WeldCell(cfg, seed=0), [lambda g: g.t, lambda g: g.s])
        assert np.all(np.diff(r[:, 1]) >= 0.0)
        assert np.allclose(np.diff(r[:, 0]), cfg.sim.dt, rtol=1e-9, atol=1e-12)

    def test_travel_speed_command_is_obeyed_and_rate_limited(self):
        cfg = _short_cfg()
        cell = WeldCell(cfg, seed=0)
        cmd = TorchCommand.from_config(cfg)
        cell.command(
            TorchCommand(
                I_set=cmd.I_set,
                v_wire_set=cmd.v_wire_set,
                arc_len_set=cmd.arc_len_set,
                v_travel=9.0e-3,
                ctwd=cmd.ctwd,
            )
        )
        speeds = [cell.step().v_travel for _ in range(int(0.5 / cfg.sim.dt))]
        assert max(np.diff(speeds)) <= cfg.robot.a_travel_max * cfg.sim.dt + 1e-12
        assert speeds[-1] == pytest.approx(9.0e-3, rel=0.02)

    def test_weave_command_produces_a_bounded_oscillation(self):
        cfg = _short_cfg()
        cell = WeldCell(cfg, seed=0)
        cmd = TorchCommand.from_config(cfg)
        cell.command(
            TorchCommand(
                I_set=cmd.I_set,
                v_wire_set=cmd.v_wire_set,
                arc_len_set=cmd.arc_len_set,
                v_travel=cmd.v_travel,
                weave_amp=2.0e-3,
                ctwd=cmd.ctwd,
            )
        )
        offs = [cell.step().weave_offset for _ in range(int(1.5 / cfg.sim.dt))]
        assert max(offs) == pytest.approx(2.0e-3, rel=0.05)
        assert min(offs) == pytest.approx(-2.0e-3, rel=0.05)


class TestWeldCellPhysicsIntegration:
    def test_no_nan_anywhere(self):
        keys = [
            lambda g: g.pool.T_pool, lambda g: g.pool.w, lambda g: g.pool.p,
            lambda g: g.pool.f, lambda g: g.arc.V, lambda g: g.arc.I,
            lambda g: g.stickout, lambda g: g.arc_length, lambda g: g.smoke,
            lambda g: g.torch_force,
        ]
        r = _collect(WeldCell(_short_cfg(), seed=0), keys)
        assert np.all(np.isfinite(r))

    def test_self_regulation_pins_the_mean_current_to_the_burn_off_law(self):
        """Wire fed must equal wire melted, so the mean current is set by the
        wire feed speed — not by the operator's current dial.  This is why DC
        current alone cannot report penetration."""
        cfg = _short_cfg()
        cell = WeldCell(cfg, seed=0)
        rows = [
            (g.arc.I, g.stickout, g.v_wire, g.arc.in_short) for g in cell.run()
        ]
        arr = np.array([(a, b, c) for a, b, c, short in rows if not short])
        I_mean, so_mean, vw_mean = arr.mean(axis=0)
        assert melting_rate(I_mean, so_mean, cfg.arc) == pytest.approx(vw_mean, rel=0.05)

    def test_constant_voltage_machine_holds_its_voltage(self):
        cfg = _short_cfg()
        cell = WeldCell(cfg, seed=0)
        v = np.array([g.arc.V for g in cell.run() if not g.arc.in_short])
        assert v.std() / v.mean() < 0.05

    def test_pool_oscillation_is_visible_in_the_voltage_spectrum(self):
        """The core sensing claim: the pool ripple must survive into V(t)."""
        cfg = _short_cfg()
        cell = WeldCell(cfg, seed=0)
        rows = [(g.arc.V, g.arc.f_osc, g.arc.in_short) for g in cell.run()]
        arr = np.array(rows, dtype=float)
        n = 8192
        seg = arr[-n:, 0].copy()
        shorted = arr[-n:, 2] > 0.5
        f_true = float(np.median(arr[-n:, 1]))
        # Blank the short-circuit periods: a 20 V collapse would otherwise bury
        # a 0.4 V ripple.  Real feature extraction does exactly this first.
        idx = np.arange(n)
        seg[shorted] = np.interp(idx[shorted], idx[~shorted], seg[~shorted])
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, cfg.sim.dt)
        band = (freqs > 20.0) & (freqs < 400.0)
        f_peak = float(freqs[band][np.argmax(spec[band])])
        assert f_peak == pytest.approx(f_true, rel=0.20)

    def test_short_circuits_collapse_the_voltage(self):
        cfg = _short_cfg()
        cell = WeldCell(cfg, seed=0)
        rows = np.array([(g.arc.V, g.arc.in_short) for g in cell.run()])
        shorted = rows[rows[:, 1] > 0.5, 0]
        arcing = rows[rows[:, 1] < 0.5, 0]
        assert len(shorted) > 0
        assert shorted.max() <= cfg.arc.V_short + 1e-9
        assert arcing.mean() > 3.0 * shorted.mean()

    def test_smoke_grows_with_arc_power_and_stays_positive(self):
        r = _collect(
            WeldCell(_short_cfg(), seed=0),
            [lambda g: g.smoke, lambda g: g.derived.q_arc],
        )
        assert np.all(r[:, 0] >= 0.0)
        assert np.corrcoef(r[:, 0], r[:, 1])[0, 1] > 0.1

    def test_cold_start_ignites_and_reaches_a_steady_bead(self):
        cfg = _short_cfg()
        cell = WeldCell(cfg, seed=0)
        cell.start_cold()
        ps = [cell.step().pool.p for _ in range(int(8.0 / cfg.sim.dt))]
        assert ps[0] < 1.0e-3
        assert ps[-1] > 2.5e-3
        assert np.std(ps[-2000:]) < 2.0e-4


class TestWeldCellVariableGap:
    """The pain point the whole proposal is about."""

    def test_penetration_tracks_the_gap_profile(self):
        cfg = default_config(seam__kind="ramp", seam__length=0.12)
        r = _collect(
            WeldCell(cfg, seed=0), [lambda g: g.gap, lambda g: g.pool.p]
        )
        assert np.corrcoef(r[:, 0], r[:, 1])[0, 1] > 0.8

    def test_fixed_parameters_burn_through_on_the_wide_gap_section(self):
        """Baseline settings that are correct at a tight fit-up must fail when
        the gap opens.  If this ever stops being true the demo has no premise."""
        cfg = default_config(seam__kind="step", seam__length=0.20)
        r = _collect(
            WeldCell(cfg, seed=0),
            [lambda g: g.gap, lambda g: float(g.defects.burn_through)],
        )
        wide = r[:, 0] > 3.5e-3
        tight = r[:, 0] < 1.0e-3
        assert wide.sum() > 0 and tight.sum() > 0
        assert r[wide, 1].mean() > 0.5
        assert r[tight, 1].mean() == 0.0
