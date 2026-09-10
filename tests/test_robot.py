"""Tests for the manipulator kinematics used by the third-person render.

The arm is a visualisation, but its kinematics is either right or it isn't,
and "the commanded torch path is reachable in one posture without hitting a
joint limit" is a real thing to be able to state.  So: exact FK/IK round-trip,
honest failure outside the workspace, and a reachability report over the
actual seam.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from weldloop.viz.robot import (
    ArmGeometry,
    Unreachable,
    fk,
    ik,
    joint_names,
    solve_path,
    tcp_pose,
    within_limits,
)


@pytest.fixture(scope="module")
def geom():
    return ArmGeometry()


def _seam_path(n: int = 25, length: float = 0.20, z: float = 0.020):
    pts, rots = [], []
    for x in np.linspace(0.0, length, n):
        p, R = tcp_pose(float(x), 0.0, z)
        pts.append(p)
        rots.append(R)
    return np.asarray(pts), rots


class TestTcpPose:
    def test_orientation_is_a_proper_rotation(self):
        _, R = tcp_pose(0.1, 0.0, 0.02)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_the_torch_points_into_the_plate(self):
        _, R = tcp_pose(0.1, 0.0, 0.02, drag_angle=0.0)
        assert R[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)

    def test_drag_angle_tilts_the_torch_back_along_travel(self):
        _, R = tcp_pose(0.1, 0.0, 0.02, drag_angle=0.30)
        approach = R[:, 2]
        assert approach[0] < 0.0            # leaning back against +x travel
        assert approach[2] < 0.0            # still pointing down
        assert math.acos(-approach[2]) == pytest.approx(0.30, abs=1e-9)


class TestForwardInverse:
    def test_round_trip_is_exact_along_the_whole_seam(self, geom):
        pts, rots = _seam_path()
        for p, R in zip(pts, rots):
            q = ik(p, R, geom)
            joints, R_fk = fk(q, geom)
            assert joints[-1] == pytest.approx(p, abs=1e-12)
            assert np.allclose(R_fk, R, atol=1e-11)

    def test_link_lengths_are_respected(self, geom):
        pts, rots = _seam_path(n=5)
        q = ik(pts[2], rots[2], geom)
        joints, _ = fk(q, geom)
        assert np.linalg.norm(joints[2] - joints[1]) == pytest.approx(geom.upper_arm)
        assert np.linalg.norm(joints[3] - joints[2]) == pytest.approx(geom.forearm)
        assert np.linalg.norm(joints[4] - joints[3]) == pytest.approx(geom.tool_length)

    def test_the_base_stays_on_the_floor_where_it_was_put(self, geom):
        pts, rots = _seam_path(n=3)
        joints, _ = fk(ik(pts[1], rots[1], geom), geom)
        assert joints[0] == pytest.approx([geom.base_xy[0], geom.base_xy[1], 0.0])

    def test_elbow_up_and_down_are_different_but_both_valid(self, geom):
        pts, rots = _seam_path(n=3)
        up = ik(pts[1], rots[1], geom, elbow_up=True)
        down = ik(pts[1], rots[1], geom, elbow_up=False)
        assert not np.allclose(up, down)
        for q in (up, down):
            assert fk(q, geom)[0][-1] == pytest.approx(pts[1], abs=1e-12)

    def test_out_of_reach_raises_rather_than_returning_nonsense(self, geom):
        p, R = tcp_pose(5.0, 0.0, 0.02)
        with pytest.raises(Unreachable):
            ik(p, R, geom)

    def test_too_close_also_raises(self, geom):
        """Inside the inner workspace boundary is just as impossible.

        The inner boundary is ``|a2 - a3|`` from the shoulder, so the wrist
        centre is placed inside that radius and the pose is built backwards
        from it.
        """
        inner = abs(geom.upper_arm - geom.forearm)
        _, R = tcp_pose(0.1, 0.0, 0.02)
        wrist = np.array([
            geom.base_xy[0] + geom.shoulder_offset + 0.4 * inner,
            geom.base_xy[1],
            geom.base_height,
        ])
        p = wrist + geom.tool_length * R[:, 2]
        with pytest.raises(Unreachable):
            ik(p, R, geom)


class TestTaskPlanner:
    """The planner is a stub, but its contract is load-bearing: it must produce
    a starting point and be honest that that is all it is."""

    def test_a_fit_up_scan_changes_the_plan(self):
        from weldloop.config import default_config
        from weldloop.planning.task_planner import SeamDescription, TaskPlanner
        from weldloop.sim.seam import make_seam

        cfg = default_config(seam__kind="step", seam__length=0.20)
        desc = SeamDescription(length=0.20, thickness=cfg.joint.thickness)
        planner = TaskPlanner(cfg)
        blind = planner.plan(desc)
        seeing = planner.plan(desc, measured=make_seam(cfg.seam, 0))
        assert {s.gap_class for s in blind.segments} == {"nominal"}
        assert {s.gap_class for s in seeing.segments} == {"tight", "nominal", "wide"}
        # a wider gap must get less current and more weave
        wide = next(s for s in seeing.segments if s.gap_class == "wide")
        tight = next(s for s in seeing.segments if s.gap_class == "tight")
        assert wide.I_set < tight.I_set
        assert wide.weave_amp > tight.weave_amp
        assert wide.v_travel < tight.v_travel

    def test_the_plan_states_what_it_does_not_know(self):
        from weldloop.planning.task_planner import SeamDescription, TaskPlanner

        plan = TaskPlanner().plan(SeamDescription(length=0.2, thickness=0.006))
        assert plan.assumptions
        assert plan.p_lo < plan.p_target < plan.p_hi
        assert plan.estimated_cycle_time > 0.0

    def test_segments_tile_the_seam_without_gaps(self):
        from weldloop.planning.task_planner import SeamDescription, TaskPlanner

        plan = TaskPlanner().plan(SeamDescription(length=0.2, thickness=0.006), n_segments=5)
        assert plan.segments[0].s_start == pytest.approx(0.0)
        assert plan.segments[-1].s_end == pytest.approx(0.2)
        for a, b in zip(plan.segments, plan.segments[1:]):
            assert a.s_end == pytest.approx(b.s_start)


class TestReachabilityOverTheSeam:
    def test_the_commanded_path_is_reachable_in_one_posture(self, geom):
        pts, rots = _seam_path(n=60)
        Q, report = solve_path(pts, rots, geom)
        assert report["unreachable_poses"] == 0
        assert all(within_limits(q, geom) for q in Q)

    def test_there_is_real_margin_on_every_joint(self, geom):
        pts, rots = _seam_path(n=60)
        _, report = solve_path(pts, rots, geom)
        assert report["worst_joint_margin_rad"] > math.radians(20.0)
        assert report["worst_joint"] in joint_names

    def test_the_wrist_never_passes_through_its_singularity(self, geom):
        """q5 near zero is where the ZYZ wrist loses a degree of freedom and the
        rendered arm would snap; the posture must stay clear of it."""
        pts, rots = _seam_path(n=60)
        _, report = solve_path(pts, rots, geom)
        assert report["min_wrist_bend_rad"] > math.radians(5.0)

    def test_joint_motion_along_the_seam_is_smooth(self, geom):
        """No branch flips and no 2*pi wraps: consecutive poses must not jump.

        Without the unwrap in ``solve_path`` the wrist roll comes back from
        atan2 with a 2*pi discontinuity partway along the seam, and the
        rendered arm snaps.
        """
        pts, rots = _seam_path(n=120)
        Q, _ = solve_path(pts, rots, geom)
        assert np.max(np.abs(np.diff(Q, axis=0))) < math.radians(3.0)

    def test_weaving_stays_inside_the_workspace(self, geom):
        """The torch also moves across the seam; that must not break reach."""
        pts, rots = [], []
        for x in np.linspace(0.0, 0.20, 60):
            y = 0.004 * math.sin(40.0 * x)
            p, R = tcp_pose(float(x), float(y), 0.020)
            pts.append(p)
            rots.append(R)
        _, report = solve_path(np.asarray(pts), rots, geom)
        assert report["unreachable_poses"] == 0
