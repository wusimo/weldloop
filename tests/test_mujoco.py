"""Tests for the MuJoCo cell: a real UR10e standing in for the travel servo.

The point of these is not that MuJoCo works — it is that the swap is *sound*:
the arm tracks the commanded path closely enough that the weld physics sees
essentially the motion it was promised, the adapter satisfies the same
interface as the simple servo, and the Phase 4 conclusion survives being run
on an articulated arm rather than on a point that moves at a commanded speed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from weldloop.config import default_config
from weldloop.interfaces import RobotBase
from weldloop.sim.mujoco_cell import CellLayout, mujoco_available
from weldloop.sim.seam import make_seam

pytestmark = pytest.mark.skipif(not mujoco_available(), reason="mujoco not installed")


def _cfg(**over):
    over.setdefault("seam__kind", "step")
    over.setdefault("seam__length", 0.20)
    return default_config(**over)


@pytest.fixture(scope="module")
def cfg_seam():
    cfg = _cfg()
    return cfg, make_seam(cfg.seam, 0)


@pytest.fixture(scope="module")
def robot(cfg_seam):
    from weldloop.sim.mujoco_cell import MujocoRobot

    cfg, seam = cfg_seam
    return MujocoRobot(cfg, seam)


# ==========================================================================
class TestCellLayout:
    def test_seam_point_and_project_are_inverses(self):
        lay = CellLayout()
        for s, lat in ((0.0, 0.0), (0.12, 0.003), (0.2, -0.004)):
            p = lay.seam_point(s, lat)
            s_hat, lat_hat, h = lay.project(p)
            assert s_hat == pytest.approx(s, abs=1e-12)
            assert lat_hat == pytest.approx(lat, abs=1e-12)
            assert h == pytest.approx(0.0, abs=1e-12)

    def test_the_seam_sits_on_top_of_the_plate(self):
        lay = CellLayout()
        assert lay.plate_top == pytest.approx(lay.table_top + lay.plate_thickness)
        assert lay.seam_point(0.0)[2] == pytest.approx(lay.plate_top)


class TestCellModel:
    def test_it_compiles_with_the_pieces_the_render_needs(self, cfg_seam):
        import mujoco

        from weldloop.sim.mujoco_cell import build_cell

        cfg, seam = cfg_seam
        model, layout = build_cell(cfg, seam)
        names = lambda kind, n: [  # noqa: E731
            mujoco.mj_id2name(model, kind, i) for i in range(n)
        ]
        assert "torch" in names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
        assert "tcp" in names(mujoco.mjtObj.mjOBJ_SITE, model.nsite)
        assert {"cell", "closeup", "over_shoulder"} <= set(
            names(mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
        )
        assert model.nu == 6
        beads = [n for n in names(mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
                 if n and n.startswith("bead_")]
        assert len(beads) > 100

    def test_the_robot_stands_on_its_pedestal(self, cfg_seam):
        import mujoco

        from weldloop.sim.mujoco_cell import build_cell

        cfg, seam = cfg_seam
        model, layout = build_cell(cfg, seam)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        assert model.body_pos[bid][2] == pytest.approx(layout.pedestal_height)

    def test_gravity_compensation_is_on_for_the_arm(self, cfg_seam):
        import mujoco

        from weldloop.sim.mujoco_cell import build_cell

        cfg, seam = cfg_seam
        model, _ = build_cell(cfg, seam)
        for link in ("shoulder_link", "forearm_link", "wrist_3_link"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            assert model.body_gravcomp[bid] == pytest.approx(1.0)


class TestRobotAdapter:
    def test_it_is_a_robot_base(self, robot):
        assert isinstance(robot, RobotBase)
        for attr in ("s", "v_travel", "weave_amp", "weave_offset", "ctwd"):
            assert hasattr(robot, attr), attr

    def test_the_warm_start_puts_the_tcp_on_the_seam(self, robot):
        assert robot.s == pytest.approx(0.0, abs=1.5e-3)
        assert robot.tracking_error < 1.0e-3

    def test_joints_stay_inside_their_limits(self, robot):
        m = robot.model
        assert np.all(robot.qpos >= m.jnt_range[:6, 0] - 1e-9)
        assert np.all(robot.qpos <= m.jnt_range[:6, 1] + 1e-9)


class TestTracking:
    @pytest.fixture(scope="class")
    def run(self, cfg_seam):
        from weldloop.sim.mujoco_cell import MujocoRobot

        cfg, seam = cfg_seam
        rb = MujocoRobot(cfg, seam)
        rb.command(v_travel=4.5e-3, weave_amp=2.0e-3, torch_angle=0.0,
                   ctwd=cfg.robot.ctwd_nom)
        err, weave, speed = [], [], []
        for i in range(int(8.0 / cfg.sim.dt)):
            rb.step(cfg.sim.dt)
            if i > int(2.5 / cfg.sim.dt):
                err.append(rb.tracking_error)
                weave.append(rb.weave_offset)
                speed.append(rb.v_travel)
        return rb, np.array(err), np.array(weave), np.array(speed)

    def test_tcp_tracking_error_is_sub_millimetre(self, run):
        _, err, _, _ = run
        assert err.mean() < 0.5e-3
        assert err.max() < 1.5e-3

    def test_travel_speed_is_delivered(self, run):
        _, _, _, speed = run
        assert speed.mean() == pytest.approx(4.5e-3, rel=0.02)

    def test_the_weave_is_delivered_at_the_commanded_amplitude(self, run):
        """A 2 Hz weave is the fastest thing the motion layer asks for; if the
        arm could not follow it, the whole weave strategy would be fiction."""
        _, _, weave, _ = run
        assert np.max(np.abs(weave)) == pytest.approx(2.0e-3, rel=0.20)

    def test_the_commanded_path_does_not_run_away_from_the_arm(self, run):
        rb, _, _, _ = run
        assert abs(rb.s_cmd - rb.s) < 1.0e-3

    def test_it_is_deterministic(self, cfg_seam):
        from weldloop.sim.mujoco_cell import MujocoRobot

        cfg, seam = cfg_seam
        out = []
        for _ in range(2):
            rb = MujocoRobot(cfg, seam)
            rb.command(v_travel=6.0e-3, weave_amp=1.5e-3, torch_angle=0.0,
                       ctwd=cfg.robot.ctwd_nom)
            for _ in range(2000):
                rb.step(cfg.sim.dt)
            out.append(np.array([rb.s, rb.weave_offset, *rb.qpos]))
        assert np.array_equal(out[0], out[1])


@pytest.mark.slow
class TestConclusionSurvivesARealArm:
    """The claim that matters: swapping in an articulated arm does not rescue
    the fixed-parameter weld, and does not spoil the adaptive one."""

    @pytest.fixture(scope="class")
    def arms(self):
        from weldloop.control.baseline import BaselineController
        from weldloop.control.motion_layer import AdaptiveController
        from weldloop.metrics import score
        from weldloop.sim.mujoco_cell import MujocoRobot
        from weldloop.sim.runner import simulate

        cfg = _cfg(seam__length=0.14)
        seam = make_seam(cfg.seam, 0)
        out = {}
        for name, ctl in (
            ("baseline", BaselineController(cfg)),
            ("adaptive", AdaptiveController(cfg)),
        ):
            rb = MujocoRobot(cfg, seam)
            table = simulate(cfg, seed=0, seam=seam, controller=ctl, robot=rb).table
            out[name] = (score(table, cfg, name), rb)
        return cfg, out

    def test_fixed_parameters_still_burn_through(self, arms):
        _, out = arms
        assert out["baseline"][0].burn_through_length_mm > 1.0

    def test_the_adaptive_controller_still_does_not(self, arms):
        _, out = arms
        base, adap = out["baseline"][0], out["adaptive"][0]
        assert adap.burn_through_length_mm < 0.25 * base.burn_through_length_mm
        assert adap.penetration_std_mm < 0.7 * base.penetration_std_mm

    def test_the_arm_tracked_throughout(self, arms):
        _, out = arms
        for name, (_, rb) in out.items():
            assert rb.tracking_error < 1.0e-3, name
