"""A real welding cell in MuJoCo: UR10e, torch, fixture, workpiece.

Why this exists
---------------
Everything else in this repository treats the manipulator as a travel-speed
servo, because that is all the weld physics needs.  That is defensible, but it
leaves two questions unanswered that a customer will ask:

1. *Can a real arm actually run this motion?*  The adaptive controller changes
   travel speed and weave amplitude continuously; a 2 Hz weave with a
   millimetre-scale amplitude on top of a few-mm/s traverse is not obviously
   free.
2. *What does the tracking error do to the weld?*  A commanded travel speed is
   not an achieved one.

So this module replaces ``SimRobot`` with a genuine articulated arm — the
Universal Robots UR10e from MuJoCo Menagerie, with real link inertias, joint
limits and the PD position actuators shipped with that model — carrying a
torch, over a clamped workpiece.  It implements the same ``RobotBase``
interface, which is the entire point of that interface existing:

* the **commanded** path parameter is integrated from the controller's travel
  speed, exactly as before;
* a damped-least-squares differential IK turns the resulting TCP pose into
  joint targets;
* MuJoCo integrates the arm under its own actuator dynamics;
* the **achieved** TCP is then measured back off the model and projected onto
  the seam, and *that* is what the weld physics sees.

Tracking error, servo lag and weave attenuation therefore reach the melt pool.
Nothing else in the repository changes.

TODO(real-hw): this is one step from the real thing.  Replace the MuJoCo step
with an EGM/RSI stream to the controller and the measured TCP with the robot's
own feedback, and the adapter above it is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from weldloop._fastmath import clip
from weldloop.config import WeldConfig
from weldloop.interfaces import RobotBase
from weldloop.sim.seam import Seam

__all__ = ["CellLayout", "build_cell", "MujocoRobot", "mujoco_available"]


def mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True, slots=True)
class CellLayout:
    """Where everything sits in the cell.  Metres, world frame, Z up."""

    table_top: float = 0.60
    plate_thickness: float = 0.006
    seam_x: float = 0.64
    seam_y0: float = -0.10
    seam_dir: tuple[float, float, float] = (0.0, 1.0, 0.0)
    plate_half_width: float = 0.075
    plate_margin: float = 0.045
    torch_length: float = 0.235
    torch_bend_deg: float = 42.0
    pedestal_height: float = 0.42

    @property
    def plate_top(self) -> float:
        return self.table_top + self.plate_thickness

    def seam_point(self, s: float, lateral: float = 0.0) -> np.ndarray:
        """World point at arc length ``s`` along the seam, offset across it."""
        d = np.asarray(self.seam_dir, dtype=float)
        across = np.cross([0.0, 0.0, 1.0], d)
        return (
            np.array([self.seam_x, self.seam_y0, self.plate_top])
            + d * s
            + across * lateral
        )

    def project(self, p: np.ndarray) -> tuple[float, float, float]:
        """Inverse of :meth:`seam_point`: (s, lateral, height above the plate)."""
        d = np.asarray(self.seam_dir, dtype=float)
        across = np.cross([0.0, 0.0, 1.0], d)
        rel = np.asarray(p, dtype=float) - np.array(
            [self.seam_x, self.seam_y0, self.plate_top]
        )
        return float(rel @ d), float(rel @ across), float(rel[2])


# --------------------------------------------------------------------------
def build_cell(
    cfg: WeldConfig,
    seam: Seam,
    layout: CellLayout | None = None,
    *,
    servo_kp: float = 14000.0,
    servo_kd: float = 220.0,
    n_bead_segments: int = 220,
):
    """Build the MuJoCo model of the cell.  Returns ``(model, layout)``.

    The UR10e comes from MuJoCo Menagerie unmodified; the table, fixture,
    workpiece and torch are added around it with ``MjSpec`` so there is no
    forked copy of somebody else's robot description to keep in sync.
    """
    import mujoco
    from robot_descriptions import ur10e_mj_description

    layout = layout or CellLayout()
    spec = mujoco.MjSpec.from_file(str(ur10e_mj_description.MJCF_PATH))
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    world = spec.worldbody
    # stand the robot on its pedestal
    spec.body("base").pos = [0.0, 0.0, layout.pedestal_height]
    # Gravity compensation, as every industrial controller does.  Without it
    # the shipped PD position actuators droop about a centimetre under the
    # torch, which is a property of the model's tuning rather than of any real
    # robot, and it would masquerade as a control result.
    for link in ("shoulder_link", "upper_arm_link", "forearm_link",
                 "wrist_1_link", "wrist_2_link", "wrist_3_link"):
        spec.body(link).gravcomp = 1.0

    # Servo tuning.  Menagerie ships kp=5000, kd=500, whose slow pole sits at
    # kp/kd = 10 rad/s ~ 1.6 Hz -- directly on top of the 2 Hz weave, which
    # makes the arm resonate and is a property of a generic manipulation tuning
    # rather than of the hardware.  A welding cell is commissioned stiffer.
    # CONFIGURABLE; ``servo_kp``/``servo_kd`` below are the only numbers here
    # that differ from the published model.
    for act in spec.actuators:
        act.gainprm[0] = servo_kp
        act.biasprm[1] = -servo_kp
        act.biasprm[2] = -servo_kd

    # --- lighting and the room ------------------------------------------
    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300, height=300, rgb1=[0.115, 0.122, 0.135], rgb2=[0.135, 0.143, 0.156],
    )
    spec.add_material(name="floor", textures=["", "grid"], texrepeat=[6, 6],
                      texuniform=True, reflectance=0.05)
    spec.add_material(name="steel", rgba=[0.300, 0.322, 0.352, 1.0],
                      specular=0.26, shininess=0.38, reflectance=0.03)
    spec.add_material(name="bench", rgba=[0.175, 0.190, 0.210, 1.0],
                      specular=0.18, shininess=0.25)
    spec.add_material(name="clamp", rgba=[0.30, 0.315, 0.335, 1.0],
                      specular=0.45, shininess=0.5)
    spec.add_material(name="wall", rgba=[0.105, 0.112, 0.125, 1.0],
                      specular=0.05, shininess=0.05)
    spec.add_material(name="torch", rgba=[0.09, 0.09, 0.10, 1.0],
                      specular=0.4, shininess=0.5)
    spec.add_material(name="copper", rgba=[0.50, 0.33, 0.18, 1.0],
                      specular=0.75, shininess=0.75)
    spec.add_material(name="bead", rgba=[0.42, 0.30, 0.22, 1.0],
                      specular=0.3, shininess=0.3)
    spec.add_material(name="arcglow", rgba=[1.0, 0.97, 0.85, 1.0],
                      emission=1.0, specular=0.0, shininess=0.0)

    world.add_light(
        pos=[0.9, -0.9, 2.2], dir=[-0.35, 0.35, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=60.0, exponent=8.0,
        diffuse=[0.50, 0.505, 0.525], specular=[0.18, 0.18, 0.19], castshadow=1,
    )
    world.add_light(
        pos=[-0.8, 0.9, 1.9], dir=[0.4, -0.4, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=70.0, exponent=6.0,
        diffuse=[0.30, 0.31, 0.34], specular=[0.10, 0.10, 0.10], castshadow=0,
    )

    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4.0, 4.0, 0.1],
                   material="floor", name="floor")
    for name, pos, size in (
        ("wall_back", [0.15, 1.35, 1.15], [2.4, 0.03, 1.15]),
        ("wall_side", [-1.35, 0.10, 1.15], [0.03, 2.0, 1.15]),
    ):
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, name=name, pos=pos,
                       size=size, material="wall", contype=0, conaffinity=0)

    # --- pedestal under the robot ---------------------------------------
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, name="pedestal",
        pos=[0.0, 0.0, layout.pedestal_height / 2.0],
        size=[0.13, layout.pedestal_height / 2.0, 0.0], material="bench",
    )

    # --- welding bench ---------------------------------------------------
    bx0, bx1 = layout.seam_x - 0.26, layout.seam_x + 0.30
    by0, by1 = layout.seam_y0 - 0.22, layout.seam_y0 + 0.42
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, name="bench_top",
        pos=[(bx0 + bx1) / 2, (by0 + by1) / 2, layout.table_top - 0.018],
        size=[(bx1 - bx0) / 2, (by1 - by0) / 2, 0.018], material="bench",
    )
    for lx in (bx0 + 0.05, bx1 - 0.05):
        for ly in (by0 + 0.05, by1 - 0.05):
            world.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, name=f"leg_{lx:.2f}_{ly:.2f}",
                pos=[lx, ly, (layout.table_top - 0.036) / 2],
                size=[0.018, 0.018, (layout.table_top - 0.036) / 2], material="bench",
            )

    # --- the workpiece: two plates with the root gap between them --------
    d = np.asarray(layout.seam_dir, dtype=float)
    across = np.cross([0.0, 0.0, 1.0], d)
    length = float(seam.length)
    mean_gap = float(seam.gap.mean())
    half_t = layout.plate_thickness / 2.0
    centre = layout.seam_point(length / 2.0)
    for sign in (+1.0, -1.0):
        off = sign * (mean_gap / 2.0 + layout.plate_half_width / 2.0)
        pos = centre + across * off - np.array([0.0, 0.0, half_t])
        size = (
            np.abs(d) * (length / 2.0 + layout.plate_margin)
            + np.abs(across) * (layout.plate_half_width / 2.0)
            + np.array([0.0, 0.0, half_t])
        )
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            name=f"plate_{'a' if sign > 0 else 'b'}",
            pos=pos.tolist(), size=size.tolist(), material="steel",
        )
    # fixture clamps holding the plates down
    for frac in (0.12, 0.88):
        for sign in (+1.0, -1.0):
            p = layout.seam_point(frac * length, sign * 0.052)
            world.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, name=f"clamp_{frac:.2f}_{sign:+.0f}",
                pos=[p[0], p[1], layout.plate_top + 0.005],
                size=[0.010, 0.010, 0.005], material="clamp",
            )

    # --- the torch, on the tool flange -----------------------------------
    wrist = spec.body("wrist_3_link")
    bend = math.radians(layout.torch_bend_deg)
    torch = wrist.add_body(
        name="torch",
        pos=[0.0, 0.12, 0.0],
        quat=_quat_from_axis_angle([1.0, 0.0, 0.0], -math.pi / 2 + bend),
    )
    torch.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, name="torch_body",
        pos=[0.0, 0.0, 0.055], size=[0.026, 0.055, 0.0], material="torch",
        contype=0, conaffinity=0, mass=0.6,
    )
    torch.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, name="torch_neck",
        pos=[0.0, 0.0, 0.145], size=[0.014, 0.045, 0.0], material="torch",
        contype=0, conaffinity=0, mass=0.15,
    )
    torch.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, name="torch_nozzle",
        pos=[0.0, 0.0, 0.198], size=[0.011, 0.022, 0.0], material="copper",
        contype=0, conaffinity=0, mass=0.05,
    )
    torch.add_site(name="tcp", pos=[0.0, 0.0, layout.torch_length])
    torch.add_site(name="nozzle", pos=[0.0, 0.0, 0.220])

    # --- weld bead segments, revealed as the torch passes ----------------
    # MuJoCo cannot add geoms at run time, so the bead is pre-built as a strip
    # of flat boxes with zero alpha; the renderer reveals and colours them.
    for i in range(n_bead_segments):
        frac = (i + 0.5) / n_bead_segments
        p = layout.seam_point(frac * length)
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, name=f"bead_{i:03d}",
            pos=[p[0], p[1], layout.plate_top + 0.0008],
            size=(np.abs(d) * (length / n_bead_segments / 2.0 * 1.9)
                  + np.abs(across) * 0.004
                  + np.array([0.0, 0.0, 0.0008])).tolist(),
            material="bead", rgba=[0.42, 0.30, 0.22, 0.0],
            contype=0, conaffinity=0,
        )

    # --- arc: an emissive blob plus a light that the renderer modulates ----
    arc = world.add_body(name="arc_marker", pos=layout.seam_point(0.0).tolist(),
                         mocap=True)
    arc.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, name="arc_core", size=[0.0035, 0, 0],
        rgba=[1.0, 0.97, 0.85, 1.0], contype=0, conaffinity=0, mass=0.0,
        material="arcglow",
    )
    arc.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, name="arc_halo", size=[0.010, 0, 0],
        rgba=[1.0, 0.72, 0.30, 0.30], contype=0, conaffinity=0, mass=0.0,
    )
    arc.add_light(
        pos=[0.0, 0.0, 0.035], dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=85.0, exponent=1.5,
        diffuse=[1.0, 0.72, 0.40], specular=[0.8, 0.6, 0.4], castshadow=1,
    )

    # --- cameras ----------------------------------------------------------
    world.add_camera(
        name="cell", pos=[1.05, -0.98, 1.10], fovy=52.0,
        quat=_look_at([1.05, -0.98, 1.10], [layout.seam_x - 0.10,
                                            layout.seam_y0 + 0.10,
                                            layout.plate_top + 0.14]),
    )
    world.add_camera(
        name="over_shoulder", pos=[0.58, -0.52, 1.02], fovy=48.0,
        quat=_look_at([0.72, -0.62, 1.16], [layout.seam_x, layout.seam_y0 + 0.10,
                                            layout.plate_top]),
    )
    world.add_camera(
        name="closeup", fovy=38.0,
        pos=[layout.seam_x + 0.16, layout.seam_y0 - 0.10, layout.plate_top + 0.13],
        quat=_look_at(
            [layout.seam_x + 0.16, layout.seam_y0 - 0.10, layout.plate_top + 0.13],
            [layout.seam_x, layout.seam_y0 + 0.09, layout.plate_top],
        ),
    )

    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.global_.fovy = 42.0
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8
    spec.visual.map.znear = 0.02
    spec.visual.headlight.ambient = [0.165, 0.173, 0.188]
    spec.visual.headlight.diffuse = [0.145, 0.150, 0.162]
    spec.visual.headlight.specular = [0.04, 0.04, 0.05]

    model = spec.compile()
    return model, layout


def _quat_from_axis_angle(axis, angle: float) -> list[float]:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    s = math.sin(angle / 2.0)
    return [math.cos(angle / 2.0), *(a * s)]


def _look_at(eye, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """Quaternion for a MuJoCo camera at ``eye`` looking at ``target``.

    MuJoCo cameras look down their own -Z with +Y up.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, dtype=float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return _mat_to_quat(R)


def _mat_to_quat(R: np.ndarray) -> list[float]:
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s]
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                (R[1, 2] + R[2, 1]) / s]
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s, 0.25 * s]


# --------------------------------------------------------------------------
class MujocoRobot(RobotBase):
    """UR10e carrying the torch, behind the same interface as ``SimRobot``.

    The commanded path is integrated from the controller's travel speed just as
    before; what changes is that the TCP is then delivered by an arm with mass,
    joint limits and finite-bandwidth position actuators, and the weld sees
    what the arm actually did.
    """

    def __init__(
        self,
        cfg: WeldConfig,
        seam: Seam,
        *,
        layout: CellLayout | None = None,
        ik_damping: float = 0.05,
        k_pos: float = 25.0,
        k_rot: float = 12.0,
        lead_limit: float = 0.20,
        servo_kp: float = 14000.0,
        servo_kd: float = 220.0,
        n_bead_segments: int = 220,
        record_hz: float = 0.0,
    ) -> None:
        import mujoco

        self.cfg = cfg
        self.seam = seam
        self.model, self.layout = build_cell(
            cfg, seam, layout, servo_kp=servo_kp, servo_kd=servo_kd,
            n_bead_segments=n_bead_segments,
        )
        self.data = mujoco.MjData(self.model)
        self._mj = mujoco
        self._tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._ik_damping = ik_damping
        self._k_pos = k_pos
        self._k_rot = k_rot
        self._lead_limit = lead_limit

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        cmd_v = cfg.baseline.v_travel
        self.s_cmd = 0.0
        self._v_set = cmd_v
        self._weave_amp_set = cfg.baseline.weave_amp
        self.weave_amp = cfg.baseline.weave_amp
        self.torch_angle = cfg.robot.torch_angle
        self.ctwd = cfg.robot.ctwd_nom
        self.phase = 0.0
        self.v_cmd = cmd_v

        self.s = 0.0
        self.v_travel = cmd_v
        self.weave_offset = 0.0
        self._t = 0.0
        self._t_mj = 0.0
        self._t_last_measure = 0.0
        self.tracking_error = 0.0

        #: optional (t, qpos) history for the renderer
        self.record_hz = float(record_hz)
        self.history: list[tuple[float, np.ndarray]] = []
        self._t_next_record = 0.0

        self._seed_pose()
        self._kinematic_ik(0.0)
        for _ in range(400):                       # let the servos settle
            self._servo(self.model.opt.timestep)
            self._t_mj += self.model.opt.timestep
        self._t = self._t_mj
        self._t_last_measure = self._t
        self._measure(init=True)

    # -- posture ---------------------------------------------------------
    def _seed_pose(self) -> None:
        """A sane elbow-up posture, so the IK does not have to search."""
        self.data.qpos[:6] = np.array([-1.57, -1.35, 2.05, -2.30, -1.57, 0.0])
        self.data.qvel[:] = 0.0
        self._q_cmd = np.array(self.data.qpos[:6])
        self.data.ctrl[:6] = self._q_cmd
        self._mj.mj_forward(self.model, self.data)

    def _pose_error(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and rotation error of the TCP against the target for ``s``."""
        mj, m, d = self._mj, self.model, self.data
        p_des, R_des = self._target_pose(s)
        err_p = p_des - d.site_xpos[self._tcp]
        R_now = d.site_xmat[self._tcp].reshape(3, 3)
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, (R_des @ R_now.T).flatten())
        err_r = np.zeros(3)
        mj.mju_quat2Vel(err_r, quat, 1.0)
        return err_p, err_r

    def _dls(self, task: np.ndarray) -> np.ndarray:
        """Damped least-squares joint rates for a task-space rate."""
        self._mj.mj_jacSite(
            self.model, self.data, self._jacp, self._jacr, self._tcp
        )
        J = np.vstack([self._jacp, self._jacr])[:, :6]
        lam = self._ik_damping
        return J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), task)

    def _kinematic_ik(self, s: float, iters: int = 300) -> None:
        """Place the arm exactly on the target pose without running dynamics."""
        mj, m, d = self._mj, self.model, self.data
        for _ in range(iters):
            err_p, err_r = self._pose_error(s)
            if np.linalg.norm(err_p) < 1e-6 and np.linalg.norm(err_r) < 1e-5:
                break
            dq = self._dls(np.concatenate([err_p, 0.5 * err_r]))
            d.qpos[:6] = np.clip(
                d.qpos[:6] + 0.5 * dq, m.jnt_range[:6, 0], m.jnt_range[:6, 1]
            )
            mj.mj_forward(m, d)
        d.qvel[:] = 0.0
        self._q_cmd = np.array(d.qpos[:6])
        d.ctrl[:6] = self._q_cmd
        mj.mj_forward(m, d)

    def _target_pose(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """Desired TCP position and orientation for path parameter ``s``."""
        lateral = self.weave_amp * math.sin(self.phase)
        p = self.layout.seam_point(
            clip(s, 0.0, self.seam.length), lateral
        ) + np.array([0.0, 0.0, self.ctwd])
        d = np.asarray(self.layout.seam_dir, dtype=float)
        drag = 0.30                                   # backhand tilt [rad]
        approach = -np.array([0.0, 0.0, 1.0]) * math.cos(drag) - d * math.sin(drag)
        approach /= np.linalg.norm(approach)
        x_t = d - (d @ approach) * approach
        x_t /= np.linalg.norm(x_t)
        y_t = np.cross(approach, x_t)
        return p, np.column_stack([x_t, y_t, approach])

    def _target_velocity(self) -> np.ndarray:
        """Analytic TCP velocity of the commanded path — the feed-forward term.

        Without it the weave lags: a proportional-only tracker sits behind a
        25 mm/s peak lateral velocity by v/k, which at these gains is millimetres
        and would show up as a fake weave attenuation.
        """
        d = np.asarray(self.layout.seam_dir, dtype=float)
        across = np.cross([0.0, 0.0, 1.0], d)
        w = 2.0 * math.pi * self.cfg.robot.weave_freq
        return d * self.v_cmd + across * (self.weave_amp * w * math.cos(self.phase))

    # -- one servo tick --------------------------------------------------
    def _servo(self, dt: float) -> None:
        mj, m, d = self._mj, self.model, self.data
        err_p, err_r = self._pose_error(self.s_cmd)
        task = np.concatenate([
            self._target_velocity() + self._k_pos * err_p,
            self._k_rot * err_r,
        ])
        dq = self._dls(task)
        q = self._q_cmd + dq * dt
        # anti-windup: the command may never run far ahead of the arm
        q = np.clip(q, d.qpos[:6] - self._lead_limit, d.qpos[:6] + self._lead_limit)
        self._q_cmd = np.clip(q, m.jnt_range[:6, 0], m.jnt_range[:6, 1])
        d.ctrl[:6] = self._q_cmd
        mj.mj_step(m, d)

    def _measure(self, init: bool = False) -> None:
        """Project the achieved TCP back onto the seam."""
        p = self.data.site_xpos[self._tcp]
        s_hat, lateral, _ = self.layout.project(p)
        s_hat = clip(s_hat, 0.0, self.seam.length)
        if init:
            self.s = s_hat
        else:
            dt = max(self._t - self._t_last_measure, 1e-9)
            alpha = min(dt / 0.010, 1.0)
            self.v_travel += (((s_hat - self.s) / dt) - self.v_travel) * alpha
            self.s = s_hat
        self.weave_offset = lateral
        err_p, _ = self._pose_error(self.s_cmd)
        self.tracking_error = float(np.linalg.norm(err_p))
        self._t_last_measure = self._t

    # -- RobotBase --------------------------------------------------------
    def command(
        self, *, v_travel: float, weave_amp: float, torch_angle: float, ctwd: float
    ) -> None:
        r = self.cfg.robot
        self._v_set = clip(v_travel, r.v_travel_min, r.v_travel_max)
        self._weave_amp_set = clip(weave_amp, 0.0, r.weave_amp_max)
        self.torch_angle = float(torch_angle)
        self.ctwd = float(ctwd)

    def step(self, dt: float) -> None:
        r = self.cfg.robot
        # commanded motion: identical law to SimRobot, so the comparison is fair
        dv = (self._v_set - self.v_cmd) * dt / r.tau_v
        dv = clip(dv, -r.a_travel_max * dt, r.a_travel_max * dt)
        self.v_cmd += dv
        self.weave_amp += (self._weave_amp_set - self.weave_amp) * dt / r.tau_v
        self.phase = (self.phase + 2.0 * math.pi * r.weave_freq * dt) % (2.0 * math.pi)
        self.s_cmd += self.v_cmd * dt
        self._t += dt

        h = self.model.opt.timestep
        while self._t_mj + h <= self._t + 1e-12:
            self._servo(h)
            self._t_mj += h
        self._measure()

        if self.record_hz > 0.0 and self._t >= self._t_next_record:
            self._t_next_record += 1.0 / self.record_hz
            self.history.append((self._t, np.array(self.data.qpos[:6])))

    @property
    def state(self) -> dict[str, float]:
        return {
            "s": float(self.s),
            "v_travel": float(self.v_travel),
            "weave_amp": float(self.weave_amp),
            "weave_offset": float(self.weave_offset),
            "ctwd": float(self.ctwd),
            "torch_angle": float(self.torch_angle),
            "tracking_error": float(self.tracking_error),
        }

    @property
    def qpos(self) -> np.ndarray:
        return np.array(self.data.qpos[:6])
