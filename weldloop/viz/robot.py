"""Kinematics of an arc-welding manipulator, for the third-person render.

Scope, stated plainly
---------------------
Nothing in the physics, the estimator or the controllers knows about a robot
arm.  ``RobotBase`` is a travel-speed servo and a weave oscillator; the weld is
determined by torch *position* and speed, not by how many joints carry the
torch there.  This module adds a manipulator **around** the tool-centre-point
trajectory the simulation already produces:

    s(t), weave(t), CTWD, torch angle   ->   TCP pose   ->   joint angles

The arm is therefore driven by real commanded motion, but it does not feed
back into the process model.  It is a visualisation, and it says so on the
figure.

It is not *only* decoration, though: the inverse kinematics can fail.  Solving
it over the whole seam checks that the commanded path is reachable in a single
posture without hitting a joint limit or a wrist singularity — which is a real
(if modest) thing to know before anyone builds the cell.

Convention
----------
Elbow manipulator with a spherical wrist, the standard arrangement for arc
welding robots:

* ``q1`` base yaw about world Z,
* ``q2``, ``q3`` shoulder and elbow in the vertical plane containing the arm,
* ``q4``, ``q5``, ``q6`` a ZYZ spherical wrist whose final Z is the tool
  approach axis.

Link lengths are those of a **generic small arc-welding arm** (0.64 m reach)
and are CONFIGURABLE.  They are not taken from any manufacturer's datasheet;
they were chosen so that a 200 mm seam and the arm both fit one frame at a
readable scale.  The base placement was then picked by searching for the
posture with the largest joint-limit margin over the seam — which is the
ordinary use of a kinematic model, and the reason this module has tests
rather than being pure decoration.  The first placement tried left only 6 deg
of elbow margin; the one below leaves 39 deg.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["ArmGeometry", "Unreachable", "tcp_pose", "ik", "fk", "joint_names"]

joint_names = ("base", "shoulder", "elbow", "wrist roll", "wrist bend", "tool roll")


class Unreachable(ValueError):
    """The requested TCP pose is outside the arm's workspace."""


@dataclass(frozen=True, slots=True)
class ArmGeometry:
    """Link lengths [m] and joint limits [rad].  Generic, configurable."""

    base_height: float = 0.34
    shoulder_offset: float = 0.09
    upper_arm: float = 0.34
    forearm: float = 0.30
    tool_length: float = 0.17
    base_xy: tuple[float, float] = (-0.16, -0.42)
    limits: tuple[tuple[float, float], ...] = field(
        default=(
            (-2.97, 2.97),   # base yaw
            (-1.75, 2.36),   # shoulder
            (-2.44, 1.22),   # elbow
            (-3.49, 3.49),   # wrist roll
            (-2.09, 2.09),   # wrist bend
            (-6.28, 6.28),   # tool roll
        )
    )

    @property
    def reach(self) -> float:
        return self.upper_arm + self.forearm


# --------------------------------------------------------------------------
def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def tcp_pose(
    x: float,
    y: float,
    z: float,
    *,
    travel_dir: tuple[float, float, float] = (1.0, 0.0, 0.0),
    drag_angle: float = 0.26,
    work_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """TCP position and orientation for a torch at ``(x, y, z)``.

    ``drag_angle`` is the usual 10-20 deg backhand tilt (positive = the torch
    leans back along the direction of travel); ``work_angle`` tilts it across
    the seam.  The tool Z axis is the approach direction, pointing into the
    plate.
    """
    t = np.asarray(travel_dir, dtype=float)
    t /= np.linalg.norm(t)
    up = np.array([0.0, 0.0, 1.0])
    across = np.cross(up, t)
    across /= np.linalg.norm(across)

    approach = -up
    # tilt back along travel, then across the seam
    approach = (
        math.cos(drag_angle) * approach + math.sin(drag_angle) * (-t)
    )
    approach = (
        math.cos(work_angle) * approach + math.sin(work_angle) * across
    )
    approach /= np.linalg.norm(approach)

    x_t = t - np.dot(t, approach) * approach
    x_t /= np.linalg.norm(x_t)
    y_t = np.cross(approach, x_t)
    R = np.column_stack([x_t, y_t, approach])
    return np.array([x, y, z], dtype=float), R


def ik(p: np.ndarray, R: np.ndarray, geom: ArmGeometry, elbow_up: bool = True) -> np.ndarray:
    """Joint angles for a TCP pose.  Raises :class:`Unreachable` if impossible."""
    base = np.array([geom.base_xy[0], geom.base_xy[1], 0.0])
    wrist = p - geom.tool_length * R[:, 2] - base

    q1 = math.atan2(wrist[1], wrist[0])
    r = math.hypot(wrist[0], wrist[1]) - geom.shoulder_offset
    s = wrist[2] - geom.base_height
    a2, a3 = geom.upper_arm, geom.forearm

    D = (r * r + s * s - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
    if abs(D) > 1.0:
        raise Unreachable(
            f"wrist centre at radius {math.hypot(r, s):.3f} m is outside "
            f"[{abs(a2 - a3):.3f}, {geom.reach:.3f}] m"
        )
    q3 = math.acos(max(-1.0, min(1.0, D)))
    if elbow_up:
        q3 = -q3
    q2 = math.atan2(s, r) - math.atan2(a3 * math.sin(q3), a2 + a3 * math.cos(q3))

    R03 = _frame3(q1, q2, q3)
    R36 = R03.T @ R
    # ZYZ Euler extraction
    q5 = math.atan2(math.hypot(R36[0, 2], R36[1, 2]), R36[2, 2])
    if abs(math.sin(q5)) < 1e-8:                       # wrist singularity
        q4 = 0.0
        q6 = math.atan2(-R36[0, 1], R36[0, 0])
    else:
        q4 = math.atan2(R36[1, 2], R36[0, 2])
        q6 = math.atan2(R36[2, 1], -R36[2, 0])
    return np.array([q1, q2, q3, q4, q5, q6])


def _frame3(q1: float, q2: float, q3: float) -> np.ndarray:
    """Orientation of the forearm frame: Z along the forearm."""
    c1, s1 = math.cos(q1), math.sin(q1)
    u = np.array([c1, s1, 0.0])
    z_up = np.array([0.0, 0.0, 1.0])
    ang = q2 + q3
    z3 = math.cos(ang) * u + math.sin(ang) * z_up      # forearm direction
    x3 = np.array([-s1, c1, 0.0])                      # normal to the arm plane
    y3 = np.cross(z3, x3)
    return np.column_stack([x3, y3, z3])


def fk(q: np.ndarray, geom: ArmGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Joint positions ``(5, 3)`` (base, shoulder, elbow, wrist, TCP) and R_tcp."""
    q1, q2, q3, q4, q5, q6 = q
    base = np.array([geom.base_xy[0], geom.base_xy[1], 0.0])
    c1, s1 = math.cos(q1), math.sin(q1)
    u = np.array([c1, s1, 0.0])
    z_up = np.array([0.0, 0.0, 1.0])

    shoulder = base + geom.shoulder_offset * u + geom.base_height * z_up
    elbow = shoulder + geom.upper_arm * (math.cos(q2) * u + math.sin(q2) * z_up)
    ang = q2 + q3
    wrist = elbow + geom.forearm * (math.cos(ang) * u + math.sin(ang) * z_up)

    R36 = _rot_z(q4) @ _rot_y(q5) @ _rot_z(q6)
    R = _frame3(q1, q2, q3) @ R36
    tcp = wrist + geom.tool_length * R[:, 2]
    return np.stack([base, shoulder, elbow, wrist, tcp]), R


def within_limits(q: np.ndarray, geom: ArmGeometry) -> bool:
    return all(lo <= qi <= hi for qi, (lo, hi) in zip(q, geom.limits))


def solve_path(
    points: np.ndarray,
    rotations: list[np.ndarray] | np.ndarray,
    geom: ArmGeometry,
    elbow_up: bool = True,
) -> tuple[np.ndarray, dict]:
    """Solve the whole TCP path and report whether the cell would work.

    Returns the joint trajectory and a small report: how close the arm came to
    its limits, and whether any pose was unreachable.  Running this before
    building a cell is the ordinary use of a kinematic model.
    """
    n = len(points)
    Q = np.empty((n, 6))
    unreachable = 0
    for i in range(n):
        try:
            Q[i] = ik(points[i], rotations[i], geom, elbow_up)
        except Unreachable:
            unreachable += 1
            Q[i] = Q[i - 1] if i else 0.0
    # atan2 wraps at +/-pi, so a smoothly moving wrist can come back as a jump
    # of 2*pi.  A real controller commands a continuous joint path and a
    # rendered arm that snaps looks broken, so unwrap.
    Q = np.unwrap(Q, axis=0)
    margins = []
    for j, (lo, hi) in enumerate(geom.limits):
        margins.append(float(min(Q[:, j].min() - lo, hi - Q[:, j].max())))
    return Q, {
        "unreachable_poses": unreachable,
        "worst_joint_margin_rad": float(min(margins)),
        "worst_joint": joint_names[int(np.argmin(margins))],
        "min_wrist_bend_rad": float(np.abs(Q[:, 4]).min()),
    }
