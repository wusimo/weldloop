"""Phase 4 tests: inner loop, motion layer, and the three-way controller comparison.

The integration test is the one the proposal rests on: on a variable-gap seam,
the adaptive controller must produce fewer burn-through events and lower
penetration variance than fixed parameters.  The RGB arm is there to show what
happens when the same control law is fed a blinded sensor.
"""

from __future__ import annotations

import numpy as np
import pytest

from weldloop.config import default_config
from weldloop.control.baseline import BaselineController
from weldloop.control.inner_loop import InnerLoop
from weldloop.control.motion_layer import AdaptiveController, MotionLayer
from weldloop.control.vision import VisionController
from weldloop.estimation.ekf import EKFOutput
from weldloop.metrics import score
from weldloop.physics.arc import melting_rate
from weldloop.sim.runner import simulate


def _cfg(**over):
    over.setdefault("seam__kind", "step")
    over.setdefault("seam__length", 0.14)
    return default_config(**over)


def _est(p: float, sig_p: float, *, w: float = 11.0e-3, sig_w: float = 0.4e-3,
         T: float = 2100.0, fill: float = 1.0) -> EKFOutput:
    """A hand-made filter output, so the control law can be tested alone."""
    x = np.array([T, w, p, fill])
    P = np.diag([80.0**2, sig_w**2, sig_p**2, 0.1**2])
    return EKFOutput(t=0.0, x=x, P=P)


def _settle(layer, est, *, gap=0.0, ticks=120, cfg=None):
    cfg = cfg or _cfg()
    out = None
    for _ in range(ticks):
        out = layer.update(
            cfg.control.dt, est, gap_ff=gap,
            v_wire_now=cfg.baseline.v_wire, thickness=cfg.joint.thickness,
        )
    return out


# ==========================================================================
class TestInnerLoop:
    def test_wire_feed_follows_the_burn_off_law(self):
        cfg = _cfg()
        il = InnerLoop(cfg)
        stickout = cfg.robot.ctwd_nom - cfg.arc.L_arc_ref
        for I_cmd in (180.0, 230.0, 275.0):
            il.reset()
            out = il.update(
                cfg.power_source.inner_dt, I_cmd=I_cmd, L_arc_cmd=cfg.arc.L_arc_ref,
                I_meas=I_cmd, L_arc_meas=cfg.arc.L_arc_ref,
            )
            assert out.v_wire_set == pytest.approx(
                melting_rate(I_cmd, stickout, cfg.arc), rel=1e-6
            )
            assert out.I_set == pytest.approx(I_cmd)

    def test_a_current_shortfall_raises_the_wire_feed(self):
        cfg = _cfg()
        il = InnerLoop(cfg)
        base = il.update(2e-3, I_cmd=230.0, L_arc_cmd=5e-3, I_meas=230.0, L_arc_meas=5e-3)
        il.reset()
        low = il.update(2e-3, I_cmd=230.0, L_arc_cmd=5e-3, I_meas=200.0, L_arc_meas=5e-3)
        assert low.v_wire_set > base.v_wire_set

    def test_a_long_arc_shortens_the_command(self):
        cfg = _cfg()
        il = InnerLoop(cfg)
        out = il.update(2e-3, I_cmd=230.0, L_arc_cmd=5e-3, I_meas=230.0, L_arc_meas=6.5e-3)
        assert out.arc_len_set < 5e-3

    def test_missing_measurements_make_it_coast(self):
        cfg = _cfg()
        il = InnerLoop(cfg)
        out = il.update(2e-3, I_cmd=230.0, L_arc_cmd=5e-3,
                        I_meas=float("nan"), L_arc_meas=float("nan"))
        assert np.isfinite([out.I_set, out.v_wire_set, out.arc_len_set]).all()
        assert out.arc_len_set == pytest.approx(5e-3)

    def test_outputs_respect_machine_limits(self):
        cfg = _cfg()
        il = InnerLoop(cfg)
        ps = cfg.power_source
        rng = np.random.default_rng(0)
        for _ in range(500):
            out = il.update(
                2e-3,
                I_cmd=float(rng.uniform(0.0, 600.0)),
                L_arc_cmd=float(rng.uniform(0.0, 20e-3)),
                I_meas=float(rng.uniform(0.0, 600.0)),
                L_arc_meas=float(rng.uniform(0.0, 20e-3)),
            )
            assert ps.I_min <= out.I_set <= ps.I_max
            assert ps.v_wire_min <= out.v_wire_set <= ps.v_wire_max
            assert ps.L_arc_min <= out.arc_len_set <= 12e-3


# ==========================================================================
class TestMotionLayerLaw:
    def test_a_deep_pool_gets_less_current_and_more_weave(self):
        cfg = _cfg()
        deep = _settle(MotionLayer(cfg), _est(5.2e-3, 0.2e-3))
        shallow = _settle(MotionLayer(cfg), _est(3.2e-3, 0.2e-3))
        assert deep.I_cmd < cfg.baseline.I_set < shallow.I_cmd
        assert deep.weave_amp > shallow.weave_amp

    def test_current_is_the_primary_penetration_authority(self):
        cfg = _cfg()
        out = _settle(MotionLayer(cfg), _est(3.0e-3, 0.15e-3))
        assert out.I_cmd > cfg.baseline.I_set * 1.05
        assert cfg.control.I_min_cmd <= out.I_cmd <= cfg.control.I_max_cmd

    def test_uncertainty_makes_it_conservative(self):
        """Same mean penetration, more uncertainty: the productivity push must
        disappear.  This is the requirement that the controller consume the
        covariance, not just the mean."""
        cfg = _cfg()
        sure = _settle(MotionLayer(cfg), _est(cfg.control.p_target, 0.10e-3))
        unsure = _settle(MotionLayer(cfg), _est(cfg.control.p_target, 0.60e-3))
        assert sure.v_travel > unsure.v_travel
        assert sure.slack > unsure.slack

    def test_a_wider_gap_slows_the_torch_so_the_filler_keeps_up(self):
        """The fill limit is one-sided: it does nothing until the joint needs
        more filler than the wire can lay down at the productivity speed."""
        cfg = _cfg()
        speeds = [
            _settle(MotionLayer(cfg), _est(cfg.control.p_target, 0.2e-3), gap=g).v_travel
            for g in (0.0, 4.0e-3, 6.0e-3)
        ]
        assert speeds[0] > speeds[1] > speeds[2]
        # at a tight fit-up the limit is slack, so speed is set by productivity
        assert speeds[0] > cfg.baseline.v_travel

    def test_a_wider_gap_gets_more_weave(self):
        cfg = _cfg()
        weaves = [
            _settle(MotionLayer(cfg), _est(cfg.control.p_target, 0.2e-3), gap=g).weave_amp
            for g in (0.0, 2.0e-3, 4.0e-3)
        ]
        assert weaves[0] < weaves[1] < weaves[2]

    def test_predicted_burn_through_trips_the_safety_path(self):
        cfg = _cfg()
        layer = MotionLayer(cfg)
        nominal = _settle(layer, _est(cfg.control.p_target, 0.2e-3))
        layer2 = MotionLayer(cfg)
        danger = layer2.update(
            cfg.control.dt, _est(5.7e-3, 0.3e-3), gap_ff=4.0e-3,
            v_wire_now=cfg.baseline.v_wire, thickness=cfg.joint.thickness,
        )
        assert danger.emergency and danger.reason == "predicted burn-through"
        assert danger.weave_amp == pytest.approx(cfg.robot.weave_amp_max)
        assert danger.I_cmd < nominal.I_cmd

    def test_the_safety_monitor_never_predicts_safer_than_it_believes(self):
        cfg = _cfg()
        layer = MotionLayer(cfg)
        est = _est(5.0e-3, 0.2e-3)
        out = layer.update(
            cfg.control.dt, est, gap_ff=1.0e-3,
            v_wire_now=cfg.baseline.v_wire, thickness=cfg.joint.thickness,
        )
        assert out.p_pred >= est.penetration - 1e-12

    def test_commands_stay_inside_every_limit(self):
        cfg = _cfg()
        layer = MotionLayer(cfg)
        rng = np.random.default_rng(0)
        prev_v = layer.v_cmd
        for _ in range(800):
            est = _est(
                float(rng.uniform(0.5e-3, 7.0e-3)),
                float(rng.uniform(0.02e-3, 1.5e-3)),
                w=float(rng.uniform(4e-3, 20e-3)),
                fill=float(rng.uniform(0.0, 2.5)),
            )
            out = layer.update(
                cfg.control.dt, est, gap_ff=float(rng.uniform(0.0, 6e-3)),
                v_wire_now=cfg.baseline.v_wire, thickness=cfg.joint.thickness,
            )
            assert cfg.robot.v_travel_min <= out.v_travel <= cfg.robot.v_travel_max
            assert 0.0 <= out.weave_amp <= cfg.robot.weave_amp_max
            assert cfg.control.I_min_cmd <= out.I_cmd <= cfg.control.I_max_cmd
            if not out.emergency:
                slew = cfg.control.v_slew * cfg.control.dt + 1e-12
                assert abs(out.v_travel - prev_v) <= slew
            prev_v = out.v_travel


class TestBaselineController:
    def test_it_never_changes_anything(self):
        cfg = _cfg()
        ctl = BaselineController(cfg)
        first = ctl.update(0.0, None)
        for t in (0.5, 5.0, 30.0):
            assert ctl.update(t, None) == first
        assert first.v_travel == cfg.baseline.v_travel
        assert first.I_set == cfg.baseline.I_set


# ==========================================================================
@pytest.fixture(scope="module")
def arms():
    cfg = _cfg()
    out = {}
    for name, ctl in (
        ("baseline", BaselineController(cfg)),
        ("adaptive", AdaptiveController(cfg)),
        ("vision", VisionController(cfg)),
    ):
        table = simulate(cfg, seed=0, controller=ctl).table
        out[name] = (score(table, cfg, name), table, ctl)
    return cfg, out


class TestAdaptiveBeatsBaseline:
    """The headline comparison, on a variable-gap seam."""

    def test_the_premise_holds_fixed_parameters_burn_through(self, arms):
        _, out = arms
        base = out["baseline"][0]
        assert base.burn_through_events > 0
        assert base.burn_through_length_mm > 1.0

    def test_fewer_burn_through_events(self, arms):
        _, out = arms
        base, adap = out["baseline"][0], out["adaptive"][0]
        assert adap.burn_through_events < base.burn_through_events
        assert adap.burn_through_length_mm < 0.2 * base.burn_through_length_mm

    def test_lower_penetration_variance(self, arms):
        _, out = arms
        base, adap = out["baseline"][0], out["adaptive"][0]
        assert adap.penetration_std_mm < 0.6 * base.penetration_std_mm

    def test_more_of_the_seam_inside_the_acceptance_band(self, arms):
        _, out = arms
        assert out["adaptive"][0].penetration_in_band_pct > out["baseline"][0].penetration_in_band_pct
        assert out["adaptive"][0].penetration_in_band_pct > 95.0

    def test_no_lack_of_fusion_is_traded_for_the_improvement(self, arms):
        """Cutting burn-through by simply going faster would show up here."""
        _, out = arms
        assert out["adaptive"][0].lack_of_fusion_length_mm <= max(
            out["baseline"][0].lack_of_fusion_length_mm, 2.0
        )

    def test_cycle_time_is_not_sacrificed(self, arms):
        _, out = arms
        base, adap = out["baseline"][0], out["adaptive"][0]
        assert adap.duration_s < 1.25 * base.duration_s

    def test_defect_free_length_improves(self, arms):
        _, out = arms
        assert out["adaptive"][0].defect_free_length_pct > out["baseline"][0].defect_free_length_pct


class TestVisionArmPaysForItsUncertainty:
    """Same control law, blinded sensor.  Any gap is attributable to the sensor.

    Note what this does *not* assert.  The RGB arm does not wreck the weld —
    an earlier version of these tests claimed it did, and rate-limiting the
    current command (a change made for a completely different reason) falsified
    that.  What it actually does is survive by crawling: its filter reports a
    penetration uncertainty three times larger, the motion layer's productivity
    condition therefore almost never holds, and the arm pays for its blindness
    in cycle time.  That is the honest result, and it is the one asserted here.
    """

    def test_it_uses_the_identical_control_law(self, arms):
        cfg, out = arms
        vision, adaptive = out["vision"][2], out["adaptive"][2]
        assert type(vision.motion) is type(adaptive.motion)
        assert type(vision.inner) is type(adaptive.inner)
        assert vision.sensor_set.name == "rgb"
        assert adaptive.sensor_set.name == "all"

    def test_its_filter_is_far_less_certain(self, arms):
        """The mechanism behind every other difference in this class."""
        _, out = arms
        sig = {
            n: np.mean([r["p_std"] for r in out[n][2].history])
            for n in ("adaptive", "vision")
        }
        assert sig["vision"] > 2.0 * sig["adaptive"]

    def test_it_pays_a_large_cycle_time_penalty(self, arms):
        _, out = arms
        assert out["vision"][0].duration_s > 1.35 * out["adaptive"][0].duration_s

    def test_its_penetration_control_is_worse(self, arms):
        _, out = arms
        assert out["vision"][0].penetration_std_mm > 1.8 * out["adaptive"][0].penetration_std_mm

    def test_it_leaves_lack_of_fusion_behind(self, arms):
        _, out = arms
        assert out["vision"][0].lack_of_fusion_length_mm > 2.0
        assert (
            out["vision"][0].lack_of_fusion_length_mm
            > out["adaptive"][0].lack_of_fusion_length_mm + 2.0
        )

    def test_being_slower_is_not_automatically_safer(self, arms):
        """Crawling buys it freedom from burn-through but not a better weld."""
        _, out = arms
        assert out["vision"][0].defect_free_length_pct < out["adaptive"][0].defect_free_length_pct


class TestControllerPlumbing:
    def test_the_adaptive_controller_actually_moved_its_handles(self, arms):
        _, out = arms
        table = out["adaptive"][1]
        assert np.std(table["cmd_v_travel"]) > 1.0e-4
        assert np.max(table["cmd_weave_amp"]) > 1.0e-3
        assert np.std(table["cmd_I_set"]) > 2.0

    def test_the_three_timescales_all_ran(self, arms):
        cfg, out = arms
        ctl = out["adaptive"][2]
        duration = out["adaptive"][0].duration_s
        # motion layer at control.dt
        assert len(ctl.history) == pytest.approx(duration / cfg.control.dt, rel=0.05)
        # the feature buffer only fills after one window, so the first decisions
        # are the baseline command
        assert ctl.history[0]["t"] >= cfg.estimator_window - 1e-9

    def test_it_is_deterministic(self):
        cfg = _cfg(seam__length=0.05)
        a = simulate(cfg, seed=0, controller=AdaptiveController(cfg)).table
        b = simulate(cfg, seed=0, controller=AdaptiveController(cfg)).table
        assert np.array_equal(a["truth_penetration"], b["truth_penetration"])
