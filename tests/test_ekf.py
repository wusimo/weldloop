"""Phase 3 tests: V/I feature extraction and the melt-pool EKF.

Two claims are load-bearing for the proposal and are asserted here directly:

* the information about penetration is in the **structure** of the V/I
  waveform (the pool-oscillation ripple), not in its DC levels;
* the filter's covariance is honest — it shrinks as sensors are added, and it
  grows when the only sensor is the RGB camera, so a controller consuming it
  becomes conservative in exactly the situation where it should.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from weldloop.config import default_config
from weldloop.estimation import residual as res_mod
from weldloop.estimation.ekf import PoolEKF, SensorSet
from weldloop.estimation.evaluate import extract_series, run_ekf
from weldloop.estimation.features import (
    GapTracker,
    blank_short_circuits,
    extract_features,
    ripple_peak,
)
from weldloop.sim.runner import simulate


def _cfg(**over):
    over.setdefault("seam__kind", "step")
    over.setdefault("seam__length", 0.10)
    return default_config(**over)


@pytest.fixture(scope="module")
def weld():
    cfg = _cfg()
    return cfg, simulate(cfg, seed=0).table


@pytest.fixture(scope="module")
def series(weld):
    cfg, table = weld
    return extract_series(cfg, table)


@pytest.fixture(scope="module")
def results(weld, series):
    cfg, _ = weld
    return {name: run_ekf(cfg, series, name) for name in ("vi", "vi+profiler", "all", "rgb")}


# ==========================================================================
class TestRipplePeak:
    def test_recovers_a_synthetic_sinusoid(self):
        dt = 1.0 / 5000.0
        n = 1000
        t = np.arange(n) * dt
        f, a = 117.0, 0.42
        x = a * np.sin(2 * np.pi * f * t) + 0.05 * np.random.default_rng(0).standard_normal(n)
        f_hat, a_hat, snr = ripple_peak(x, dt, 30.0, 400.0)
        assert f_hat == pytest.approx(f, rel=0.02)
        assert a_hat == pytest.approx(a, rel=0.10)
        assert snr > 10.0

    def test_rejects_a_linear_trend(self):
        """A drifting arc length must not be read as a huge low-frequency ripple."""
        dt = 1.0 / 5000.0
        n = 1000
        t = np.arange(n) * dt
        x = 3.0 * t + 0.3 * np.sin(2 * np.pi * 90.0 * t)
        f_hat, a_hat, _ = ripple_peak(x, dt, 30.0, 400.0)
        assert f_hat == pytest.approx(90.0, rel=0.03)
        assert a_hat == pytest.approx(0.3, rel=0.12)

    def test_returns_nan_on_a_useless_window(self):
        f, a, snr = ripple_peak(np.zeros(4), 1.0 / 5000.0, 30.0, 400.0)
        assert math.isnan(f) and math.isnan(a) and snr == 0.0


class TestShortCircuitBlanking:
    def test_blanking_removes_the_collapses(self):
        V = np.full(500, 26.0)
        short = np.zeros(500)
        V[200:210] = 6.0
        short[200:210] = 1.0
        out = blank_short_circuits(V, short)
        assert np.allclose(out, 26.0)
        assert V[205] == 6.0  # input untouched

    def test_skipping_blanking_destroys_the_ripple_feature(self, weld):
        """This is why the preprocessing exists, asserted rather than asserted-in-prose."""
        cfg, table = weld
        n = int(cfg.estimator_window * cfg.sensors.f_master)
        sl = slice(-n, None)
        V, I, sh = table["ps_V"][sl], table["ps_I"][sl], table["ps_short"][sl]
        assert np.count_nonzero(sh > 0.5) > 0

        good = extract_features(0.0, V, I, sh, cfg)
        dt = 1.0 / cfg.sensors.f_master
        f_raw, a_raw, snr_raw = ripple_peak(V, dt, *cfg.estimator_band)
        f_blank, a_blank, snr_blank = ripple_peak(
            blank_short_circuits(V, sh), dt, *cfg.estimator_band
        )
        assert snr_blank > snr_raw
        # the collapses masquerade as ripple and inflate the amplitude feature,
        # which the filter would read as far too much penetration
        assert a_raw > 2.0 * a_blank
        assert good.a_ripple == pytest.approx(a_blank / cfg.arc.E_a, rel=1e-9)

    def test_a_fully_shorted_window_is_marked_invalid(self, weld):
        cfg, _ = weld
        n = 1000
        feats = extract_features(0.0, np.full(n, 6.0), np.full(n, 380.0), np.ones(n), cfg)
        assert not feats.valid
        assert feats.sc_duty == 1.0


class TestFeaturesTrackThePhysics:
    def test_ripple_frequency_tracks_the_true_pool_oscillation(self, weld, series):
        cfg, table = weld
        step = int(round(cfg.ekf.dt * table.f_master))
        n_win = int(round(cfg.estimator_window * table.f_master))
        idx = np.arange(n_win, table.n_rows, step) - 1
        truth = table["truth_f_osc"][idx]
        m = np.isfinite(series.f_ripple)
        err = series.f_ripple[m] - truth[m]
        assert abs(err.mean()) < 3.0
        assert err.std() < 8.0
        assert np.corrcoef(series.f_ripple[m], truth[m])[0, 1] > 0.9

    def test_ripple_amplitude_tracks_the_true_surface_amplitude(self, weld, series):
        cfg, table = weld
        step = int(round(cfg.ekf.dt * table.f_master))
        n_win = int(round(cfg.estimator_window * table.f_master))
        idx = np.arange(n_win, table.n_rows, step) - 1
        truth = table["truth_a_osc"][idx]
        m = np.isfinite(series.a_ripple)
        gain = float(np.sum(series.a_ripple[m] * truth[m]) / np.sum(truth[m] ** 2))
        assert gain == pytest.approx(cfg.ekf.k_ripple, abs=0.06)
        assert np.corrcoef(series.a_ripple[m], truth[m])[0, 1] > 0.8

    def test_structure_beats_dc_levels(self, weld, series):
        """The headline: mean power and mean current say almost nothing about
        penetration in a constant-voltage machine; the ripple does."""
        p = series.truth_p
        m = np.isfinite(series.a_ripple) & np.isfinite(series.I)
        r_ripple = abs(np.corrcoef(series.a_ripple[m], p[m])[0, 1])
        r_power = abs(np.corrcoef((series.V * series.I)[m], p[m])[0, 1])
        r_current = abs(np.corrcoef(series.I[m], p[m])[0, 1])
        assert r_ripple > 0.8
        assert r_power < 0.6
        assert r_ripple > 2.0 * max(r_power, r_current)


class TestGapTracker:
    def test_falls_back_before_the_first_reading(self):
        gt = GapTracker(default_gap=1.5e-3)
        assert gt.gap_at(0.05) == pytest.approx(1.5e-3)

    def test_interpolates_between_readings(self):
        gt = GapTracker()
        gt.push(0.010, 1.0e-3)
        gt.push(0.020, 3.0e-3)
        assert gt.gap_at(0.015) == pytest.approx(2.0e-3)
        assert gt.gap_at(0.030) == pytest.approx(3.0e-3)  # held beyond the last

    def test_ignores_dropouts_and_out_of_order_readings(self):
        gt = GapTracker()
        gt.push(0.010, 1.0e-3)
        gt.push(float("nan"), 2.0e-3)
        gt.push(0.020, float("nan"))
        gt.push(0.005, 9.0e-3)  # behind the last station
        assert gt.n_readings == 1

    def test_look_ahead_readings_arrive_before_the_arc_needs_them(self, weld, series):
        """The profiler's value is that its reading is already in the tracker by
        the time the torch reaches that station."""
        cfg, _ = weld
        m = np.isfinite(series.gap_profiler)
        err = series.gap_profiler[m] - series.truth_gap[m]
        assert abs(err.mean()) < 0.3e-3
        assert err.std() < 0.5e-3


# ==========================================================================
class TestEKFAccuracy:
    def test_penetration_error_is_below_threshold_on_a_nominal_seam(self, results):
        r = results["all"]
        assert r.rmse_p < 0.5e-3
        assert abs(r.bias_p) < 0.5e-3

    def test_power_source_alone_already_works(self, results):
        """The proposal's core claim, as a number: no camera, no profiler."""
        assert results["vi"].rmse_p < 0.7e-3

    def test_more_sensors_never_hurt_the_penetration_estimate(self, results):
        assert (
            results["all"].rmse_p
            < results["vi+profiler"].rmse_p
            < results["vi"].rmse_p
        )

    def test_covariance_shrinks_with_more_sensors(self, results):
        """Required behaviour: the filter must *know* it knows more."""
        assert (
            np.mean(results["all"].w_std)
            < np.mean(results["vi+profiler"].w_std)
            < np.mean(results["vi"].w_std)
        )
        assert results["all"].mean_p_std < results["vi"].mean_p_std

    def test_the_covariance_is_honest(self, results):
        """Near-nominal 2-sigma coverage, on every sensor set including RGB."""
        for name, r in results.items():
            assert 0.85 <= r.coverage_2sigma <= 1.0, name

    def test_rgb_only_is_the_negative_example(self, results):
        rgb, best = results["rgb"], results["all"]
        assert rgb.rmse_p > 3.0 * best.rmse_p
        # ...and the filter says so, instead of being confidently wrong
        assert rgb.mean_p_std > 3.0 * best.mean_p_std

    def test_the_profiler_helps_as_a_model_input_not_as_a_measurement(self, weld, series):
        """It never observes the pool, yet it cuts the error — because a wrong
        gap corrupts the prediction and no innovation can undo that."""
        cfg, _ = weld
        blind = run_ekf(cfg, series, "vi")
        seeing = run_ekf(cfg, series, "vi+profiler")
        assert SensorSet.named("vi").channels == SensorSet.named("vi+profiler").channels
        assert seeing.rmse_p < blind.rmse_p
        assert seeing.mean_p_std < blind.mean_p_std


class TestRobustToLosingTheFrequencyChannel:
    """The pessimistic case the GMAW literature warns about.

    Arc-voltage pool-oscillation sensing is established for GTAW, but in GMAW
    the reported difficulty is that the voltage can lose the *frequency*
    signature of the pool while its fluctuation amplitude still tracks
    penetration.  Since ``f_ripple`` is this repo's strongest single correlate
    of penetration, the whole thesis would be fragile if the depth estimate
    depended on it.  These tests check that it does not.
    """

    def test_penetration_survives_losing_the_frequency_channel(self, weld, series):
        cfg, _ = weld
        with_f = run_ekf(cfg, series, "all")
        without_f = run_ekf(cfg, series, "all-no-f")
        assert without_f.rmse_p < 1.15 * with_f.rmse_p

    def test_the_same_holds_on_the_power_source_alone(self, weld, series):
        cfg, _ = weld
        with_f = run_ekf(cfg, series, "vi")
        without_f = run_ekf(cfg, series, "vi-no-f")
        assert without_f.rmse_p < 1.15 * with_f.rmse_p

    def test_the_frequency_channel_is_what_constrains_pool_WIDTH(self, weld, series):
        """Losing it should cost width, not depth - that is the mechanism."""
        cfg, _ = weld
        with_f = run_ekf(cfg, series, "vi")
        without_f = run_ekf(cfg, series, "vi-no-f")
        assert without_f.rmse_w > 3.0 * with_f.rmse_w
        assert np.mean(without_f.w_std) > 3.0 * np.mean(with_f.w_std)

    def test_the_filter_admits_it_lost_a_channel(self, weld, series):
        cfg, _ = weld
        with_f = run_ekf(cfg, series, "vi")
        without_f = run_ekf(cfg, series, "vi-no-f")
        assert without_f.mean_p_std > 1.8 * with_f.mean_p_std
        assert 0.85 <= without_f.coverage_2sigma <= 1.0


class TestEKFNumerics:
    def test_covariance_stays_symmetric_and_positive_semidefinite(self, weld, series):
        cfg, _ = weld
        ekf = PoolEKF(cfg, "all")
        for k in range(0, len(series), 7):
            u = np.array(
                [series.I[k], series.V[k], series.v_travel[k], series.v_wire[k],
                 series.gap_profiler[k], cfg.joint.thickness, series.weave[k]]
            )
            out = ekf.step(
                series.t[k], u,
                {
                    "f_ripple": series.f_ripple[k], "a_ripple": series.a_ripple[k],
                    "f_sc": series.f_sc[k], "ir_T_peak": series.ir_T[k],
                    "ir_pool_width": series.ir_w[k], "L_arc_est": series.L_arc_est[k],
                },
            )
            assert np.allclose(out.P, out.P.T, atol=1e-18)
            assert np.min(np.linalg.eigvalsh(out.P)) > -1e-15
            assert np.all(np.isfinite(out.x))

    def test_missing_measurements_widen_the_covariance(self, weld):
        """Compared at the two steady states, not against the initial P — a
        stable process model contracts P on its own, so 'grew from P0' would be
        the wrong thing to assert."""
        cfg, _ = weld
        u = np.array([230.0, 27.0, 4.5e-3, 6.88 / 60, 0.0, cfg.joint.thickness, 0.0])
        channels = ("f_ripple", "a_ripple", "f_sc", "ir_T_peak", "ir_pool_width")

        blind_ekf = PoolEKF(cfg, "all")
        blind = {k: math.nan for k in channels}
        for _ in range(400):
            out_blind = blind_ekf.step(0.0, u, blind)
        assert out_blind.n_used == 0

        seeing = PoolEKF(cfg, "all")
        for _ in range(400):
            # feed each channel its own prediction: zero innovation, so only the
            # information content of having a measurement at all is exercised
            meas = {c: seeing._h(seeing.x, c, {"I": u[0], "L_arc_est": 5.0e-3})
                    for c in channels}
            out_seeing = seeing.step(0.0, u, meas)
        assert out_seeing.n_used == len(channels)
        assert out_blind.P[2, 2] > out_seeing.P[2, 2]
        assert out_blind.P[1, 1] > out_seeing.P[1, 1]

    def test_is_deterministic(self, weld, series):
        cfg, _ = weld
        a = run_ekf(cfg, series, "all")
        b = run_ekf(cfg, series, "all")
        assert np.array_equal(a.p, b.p)

    def test_a_perfect_model_beats_a_mismatched_one(self, weld, series):
        """Sanity on the deliberate mismatch: it is what costs the filter its bias."""
        cfg, _ = weld
        mismatched = run_ekf(cfg, series, "all")
        ekf = PoolEKF(cfg, "all", use_true_model=True)
        assert ekf.est_cfg.pool.eta_melt == cfg.pool.eta_melt
        # run it by hand, same inputs
        p = []
        for k in range(len(series)):
            u = np.array(
                [series.I[k], series.V[k], series.v_travel[k], series.v_wire[k],
                 series.gap_profiler[k], cfg.joint.thickness, series.weave[k]]
            )
            out = ekf.step(
                series.t[k], u,
                {"f_ripple": series.f_ripple[k], "a_ripple": series.a_ripple[k],
                 "f_sc": series.f_sc[k], "ir_T_peak": series.ir_T[k],
                 "ir_pool_width": series.ir_w[k], "L_arc_est": series.L_arc_est[k]},
            )
            p.append(out.penetration)
        rmse_true = float(np.sqrt(np.mean((np.asarray(p) - series.truth_p) ** 2)))
        assert rmse_true < mismatched.rmse_p


# ==========================================================================
class TestResidual:
    def test_missing_checkpoint_degrades_gracefully(self, tmp_path):
        assert res_mod.load_residual(tmp_path / "nope.pt", quiet=True) is None

    def test_dataset_has_the_expected_shape(self, weld):
        cfg, table = weld
        ds = res_mod.build_dataset(cfg, table)
        assert ds.X.shape[1] == 11 and ds.Y.shape[1] == 4
        assert len(ds) > 100
        # the mismatched model really is wrong about penetration
        assert np.sqrt(np.mean(ds.Y[:, 2] ** 2)) > 1.0e-5

    def test_dataset_requires_ground_truth(self, weld):
        cfg, table = weld
        with pytest.raises(KeyError):
            res_mod.build_dataset(cfg, table.real_hw_view())

    @pytest.mark.skipif(not res_mod.torch_available(), reason="torch not installed")
    def test_training_reduces_the_one_step_model_error(self, weld):
        cfg, table = weld
        ds = res_mod.build_dataset(cfg, table)
        model, stats = res_mod.train(ds, epochs=60, verbose=False)
        assert stats["rms_p_after_mm"] < stats["rms_p_before_mm"]

    @pytest.mark.skipif(not res_mod.torch_available(), reason="torch not installed")
    def test_corrections_are_clipped(self, weld):
        cfg, table = weld
        ds = res_mod.build_dataset(cfg, table)
        model, _ = res_mod.train(ds, epochs=20, verbose=False)
        wild = np.array([9000.0, 0.5, 0.5, 40.0])
        u = np.array([9999.0, 99.0, 1.0, 9.0, 0.5, 0.006, 0.5])
        c = model.correction(wild, u, cfg.ekf.dt)
        assert np.all(np.abs(c) <= model.clip + 1e-12)
        assert np.all(np.isfinite(c))
