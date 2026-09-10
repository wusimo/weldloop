"""Phase 2 tests: sensor rates, timestamps, noise statistics, and the log schema.

The headline assertions are the ones the proposal depends on:

* the power source never drops out, while the RGB camera mostly does;
* the RGB pool-width reading is an order of magnitude worse than the IR one;
* the log is time-aligned on one clock, with NaN meaning "did not sample" and
  no ``truth_`` column reachable by anything that runs in real time.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataclasses import replace

from weldloop.config import default_config
from weldloop.sim.cell import WeldCell
from weldloop.sim.logger import REAL_HW_COLUMNS, SCHEMA, TRUTH_COLUMNS
from weldloop.sim.runner import simulate
from weldloop.sim.sensors import SensorSuite


def _cfg(**over):
    over.setdefault("seam__length", 0.05)
    return default_config(**over)


@pytest.fixture(scope="module")
def run():
    return simulate(_cfg(), seed=0)


@pytest.fixture(scope="module")
def stamps():
    """Raw per-sensor sample streams from a short run."""
    cfg = _cfg()
    cell = WeldCell(cfg, seed=0)
    suite = SensorSuite(cfg, cell.seam, np.random.default_rng(1234))
    out: dict[str, list] = {s.name: [] for s in suite.sensors}
    truth = []
    for gt in cell.run():
        truth.append(gt)
        for sensor in suite.sensors:
            for smp in sensor.poll(gt.t, gt):
                out[sensor.name].append(smp)
    return out, truth, cfg


@pytest.fixture(scope="module")
def one_truth():
    """A single settled ``GroundTruth``, for unit-testing sensors in isolation."""
    cell = WeldCell(_cfg(), seed=0)
    gt = None
    for _ in range(20_000):
        gt = cell.step()
    return gt


def _paired(table, meas: str, ref: str):
    a, b = table[meas], table[ref]
    m = ~np.isnan(a) & ~np.isnan(b)
    return a[m] - b[m]


# ==========================================================================
class TestRatesAndTimestamps:
    def test_every_sensor_hits_its_configured_rate(self, stamps):
        out, truth, cfg = stamps
        duration = truth[-1].t
        expected = {
            "ps": cfg.sensors.f_power,
            "prof": cfg.sensors.f_profiler,
            "ir": cfg.sensors.f_ir,
            "mic": cfg.sensors.f_mic,
            "force": cfg.sensors.f_force,
            "rgb": cfg.sensors.f_rgb,
        }
        for name, samples in out.items():
            rate = len(samples) / duration
            assert rate == pytest.approx(expected[name], rel=0.01), name

    def test_timestamps_are_monotone_and_on_the_sensors_own_grid(self, stamps):
        out, _, _ = stamps
        for name, samples in out.items():
            t = np.array([s.t for s in samples])
            assert np.all(np.diff(t) > 0.0), name
            period = np.median(np.diff(t))
            assert np.allclose(np.diff(t), period, rtol=1e-9), name

    def test_the_microphone_outruns_the_simulation_step(self, stamps):
        """20 kHz sensor against a 5 kHz sim: four samples per step."""
        out, truth, cfg = stamps
        assert len(out["mic"]) == pytest.approx(4 * len(truth), rel=0.01)

    def test_slow_sensors_return_empty_polls_most_of_the_time(self, stamps):
        out, truth, cfg = stamps
        assert len(out["ir"]) < 0.01 * len(truth)


class TestPowerSourceSensor:
    def test_noise_is_zero_mean_with_the_configured_std(self, one_truth):
        """Measured against a frozen arc state, so the pool ripple and the
        short-circuit collapses cannot be mistaken for sensor noise."""
        cfg = _cfg()
        from weldloop.sim.sensors import PowerSourceSensor

        sensor = PowerSourceSensor(cfg.sensors, np.random.default_rng(0))
        V = np.array([sensor.sample(0.0, one_truth).channels["ps_V"] for _ in range(40_000)])
        I = np.array([sensor.sample(0.0, one_truth).channels["ps_I"] for _ in range(40_000)])
        assert V.mean() == pytest.approx(one_truth.arc.V, abs=0.02)
        assert I.mean() == pytest.approx(one_truth.arc.I, abs=0.2)
        assert V.std() == pytest.approx(cfg.sensors.noise_V, rel=0.05)
        assert I.std() == pytest.approx(cfg.sensors.noise_I, rel=0.05)

    def test_readings_are_quantised_to_the_adc_resolution(self, run):
        cfg = run.cfg
        V = run.table["ps_V"]
        assert np.allclose(V / cfg.sensors.lsb_V, np.round(V / cfg.sensors.lsb_V))

    def test_it_never_drops_out(self, run):
        """The whole thesis: this is the sensor that always works."""
        for col in ("ps_V", "ps_I", "ps_short"):
            assert np.count_nonzero(np.isnan(run.table[col])) == 0

    def test_short_circuits_are_reported(self, run):
        assert 0.0 < np.mean(run.table["ps_short"]) < 0.3


class TestSeamProfiler:
    def test_it_measures_ahead_of_the_arc(self, run):
        lead = run.table["prof_lead_s"]
        s = run.table["rb_s"]
        m = ~np.isnan(lead) & ~np.isnan(s)
        delta = (lead - s)[m]
        # equal to the configured lead, except where it is clamped at the seam end
        assert np.median(delta) == pytest.approx(run.cfg.sensors.profiler_lead, rel=0.05)
        assert np.all(delta >= -1e-9)

    def test_dropout_rate_matches_the_configuration(self, run):
        valid = run.table["prof_valid"]
        valid = valid[~np.isnan(valid)]
        assert 1.0 - valid.mean() == pytest.approx(
            run.cfg.sensors.profiler_dropout, abs=0.04
        )

    def test_a_dropout_is_nan_not_a_plausible_number(self, run):
        gap, valid = run.table["prof_gap"], run.table["prof_valid"]
        dropped = (valid == 0.0)
        assert dropped.sum() > 0
        assert np.all(np.isnan(gap[dropped]))

    def test_gap_noise_matches_the_configuration(self, run):
        cfg = run.cfg
        gap, lead = run.table["prof_gap"], run.table["prof_lead_s"]
        m = ~np.isnan(gap)
        true_gap = np.interp(lead[m], run.seam.s, run.seam.gap)
        err = gap[m] - true_gap
        assert abs(err.mean()) < 0.5 * cfg.sensors.noise_gap
        assert err.std() == pytest.approx(cfg.sensors.noise_gap, rel=0.30)


class TestIRCamera:
    def test_error_is_dominated_by_smoke_not_by_sensor_noise(self, run):
        cfg = run.cfg
        err = _paired(run.table, "ir_T_peak", "truth_T_pool")
        assert err.std() > cfg.sensors.noise_T
        assert err.std() < 6.0 * cfg.sensors.noise_T

    def test_pool_width_is_usable(self, run):
        err = _paired(run.table, "ir_pool_width", "truth_pool_w")
        assert abs(err.mean()) < 1.0e-3
        assert err.std() < 1.5e-3

    def test_a_dense_plume_rejects_the_frame(self):
        """Force the smoke up and the camera must refuse, not guess."""
        cfg = _cfg(sensors__smoke_base=3.0)
        table = simulate(cfg, seed=0).table
        valid = table["ir_valid"]
        assert np.nanmean(valid) == 0.0
        assert np.all(np.isnan(table["ir_T_peak"]))


class TestArcMic:
    def test_logged_channel_is_a_proper_rms(self, run):
        p = run.table["mic_p"]
        m = ~np.isnan(p)
        assert m.all()
        assert np.all(p[m] >= 0.0)

    def test_level_grows_with_arc_power(self, one_truth):
        """Checked against a frozen state: along a real seam the arc power
        barely moves, so a correlation over one run proves nothing."""
        from weldloop.sim.sensors import ArcMic

        cfg = _cfg()
        levels = []
        for q_arc in (2000.0, 5000.0, 9000.0):
            gt = replace(one_truth, derived=replace(one_truth.derived, q_arc=q_arc))
            mic = ArcMic(cfg.sensors, np.random.default_rng(0))
            p = np.array([mic.sample(0.0, gt).channels["mic_p"] for _ in range(20_000)])
            levels.append(float(np.sqrt(np.mean(p**2))))
        assert levels[0] < levels[1] < levels[2]

    def test_tone_amplitude_grows_with_pool_oscillation(self, one_truth):
        from weldloop.sim.sensors import ArcMic

        cfg = _cfg()
        levels = []
        for a_osc in (0.15e-3, 0.30e-3, 0.50e-3):
            gt = replace(one_truth, arc=replace(one_truth.arc, a_osc=a_osc))
            mic = ArcMic(cfg.sensors, np.random.default_rng(0))
            p = np.array([mic.sample(0.0, gt).channels["mic_p"] for _ in range(20_000)])
            levels.append(float(np.sqrt(np.mean(p**2))))
        assert levels[0] < levels[1] < levels[2]

    def test_clicks_only_follow_short_circuits(self, run):
        clicks = np.nansum(run.table["mic_click"])
        shorts = run.table["ps_short"]
        edges = int(np.count_nonzero(np.diff(shorts) < 0))  # falling = re-ignition
        assert clicks > 0
        assert clicks == pytest.approx(edges, rel=0.05)

    def test_no_clicks_when_the_arc_never_shorts(self):
        cfg = _cfg(arc__f_sc_max=0.0)
        table = simulate(cfg, seed=0).table
        assert np.nansum(table["mic_click"]) == 0.0


class TestTorchForce:
    def test_force_follows_the_square_of_the_current(self, run):
        F, I = run.table["force_N"], run.table["ps_I"]
        m = ~np.isnan(F) & ~np.isnan(I)
        assert np.corrcoef(F[m], I[m] ** 2)[0, 1] > 0.8

    def test_noise_matches_the_configuration(self, run):
        F = run.table["force_N"]
        F = F[~np.isnan(F)]
        resid = np.diff(F)
        assert np.std(resid) / np.sqrt(2.0) > 0.5 * run.cfg.sensors.noise_force


class TestRGBCameraFails:
    """These tests exist to prove a negative, which is the point of the sensor."""

    def test_quality_collapses_during_welding(self, run):
        q = run.table["rgb_quality"]
        q = q[~np.isnan(q)]
        assert q.mean() < 0.2

    def test_it_is_invalid_a_large_fraction_of_the_time(self, run):
        valid = run.table["rgb_valid"]
        assert np.nanmean(valid) < 0.8

    def test_pool_width_is_an_order_of_magnitude_worse_than_infrared(self, run):
        ir = _paired(run.table, "ir_pool_width", "truth_pool_w")
        rgb = _paired(run.table, "rgb_pool_width", "truth_pool_w")
        assert rgb.std() > 5.0 * ir.std()

    def test_more_smoke_makes_it_worse_monotonically(self):
        qualities = []
        for base in (0.1, 0.35, 0.8):
            table = simulate(_cfg(sensors__smoke_base=base), seed=0).table
            q = table["rgb_quality"]
            qualities.append(float(np.nanmean(q[~np.isnan(q)])))
        assert qualities[0] > qualities[1] > qualities[2]


class TestDeterminismAndIsolation:
    def test_same_seed_gives_an_identical_log(self):
        a = simulate(_cfg(), seed=0).table
        b = simulate(_cfg(), seed=0).table
        for col in a.columns:
            assert np.array_equal(a[col], b[col], equal_nan=True), col

    def test_sensor_noise_does_not_perturb_the_plant(self):
        """Ablating a sensor must not change the weld that was made."""
        base = simulate(_cfg(), seed=0).table
        noisier = simulate(_cfg(sensors__noise_V=5.0, sensors__noise_T=200.0), seed=0).table
        for col in TRUTH_COLUMNS:
            assert np.array_equal(base[col], noisier[col], equal_nan=True), col


class TestLogSchema:
    def test_every_emitted_channel_is_declared_in_the_schema(self):
        cfg = _cfg()
        cell = WeldCell(cfg, seed=0)
        suite = SensorSuite(cfg, cell.seam, np.random.default_rng(0))
        declared = {c.name for c in SCHEMA}
        for sensor in suite.sensors:
            for col in sensor.columns:
                assert col in declared, col

    def test_schema_and_table_agree(self, run):
        assert set(run.table.columns) == {c.name for c in SCHEMA}

    def test_master_clock_is_uniform_and_complete(self, run):
        t = run.table["t"]
        assert not np.any(np.isnan(t))
        assert np.allclose(np.diff(t), 1.0 / run.table.f_master)

    def test_nan_density_matches_each_sensors_rate(self, run):
        n = run.table.n_rows
        f_master = run.table.f_master
        for col, rate in (("ir_valid", run.cfg.sensors.f_ir),
                          ("force_N", run.cfg.sensors.f_force),
                          ("ps_V", run.cfg.sensors.f_power)):
            filled = np.count_nonzero(~np.isnan(run.table[col]))
            assert filled / n == pytest.approx(min(rate / f_master, 1.0), rel=0.05), col

    def test_truth_columns_are_quarantined(self, run):
        view = run.table.real_hw_view()
        assert not any(c.startswith("truth_") for c in view.columns)
        assert set(view.columns) == set(REAL_HW_COLUMNS)

    def test_a_capture_shaped_log_can_be_produced(self):
        table = simulate(_cfg(), seed=0, include_truth=False).table
        assert set(table.columns) == set(REAL_HW_COLUMNS)

    def test_round_trips_through_a_file(self, run, tmp_path):
        path = run.table.write(tmp_path / "weld")
        assert path.exists() and path.stat().st_size > 0
        import pandas as pd

        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        assert list(df.columns) == list(run.table.columns)
        assert len(df) == run.table.n_rows


class TestObservationView:
    def test_the_controller_never_sees_ground_truth(self):
        """The estimator's input is assembled here; if a truth field ever leaks
        into it the whole demo is worthless."""
        from weldloop.sim.runner import Observation

        cfg = _cfg()
        cell = WeldCell(cfg, seed=0)
        suite = SensorSuite(cfg, cell.seam, np.random.default_rng(7))
        obs = Observation()
        for _ in range(20_000):
            gt = cell.step()
            obs.update(suite.poll(gt.t, gt), gt.t)
        assert obs.values
        assert not any(k.startswith("truth_") for k in obs.values)

    def test_stale_channels_report_their_age(self):
        from weldloop.sim.runner import Observation

        cfg = _cfg()
        cell = WeldCell(cfg, seed=0)
        suite = SensorSuite(cfg, cell.seam, np.random.default_rng(7))
        obs = Observation()
        for _ in range(2000):
            gt = cell.step()
            obs.update(suite.poll(gt.t, gt), gt.t)
        assert obs.age("ps_V") < 1.0e-3
        assert obs.age("ir_T_peak") < 1.0 / cfg.sensors.f_ir + 1e-3
        assert obs.age("nonexistent") == float("inf")
