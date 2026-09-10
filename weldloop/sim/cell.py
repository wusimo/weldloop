"""``WeldCell`` — the simulated welding cell.

Integrates, at the simulation rate (5 kHz by default):

* the **power source** (``SimPowerSource``): a constant-voltage inverter with
  genuine GMAW self-regulation — the electrode extension is a state, the
  current is whatever the machine's characteristic produces at the current arc
  length, and the burn-off law closes the loop;
* the **robot** (``SimRobot``): travel-speed servo, weave oscillator, torch
  height, integrating position ``s`` along the seam;
* the **arc** (``physics.arc.ArcModel``): instantaneous V/I including
  short-circuit events;
* the **melt pool** (``physics.melt_pool.MeltPoolModel``): the four-state ROM;
* slow unmodelled **disturbances** (contact-tip wear, torch-height wander,
  smoke density) as Ornstein-Uhlenbeck processes.

``step()`` is deterministic given the seed: all randomness comes from a single
master ``numpy.random.Generator`` spawned into named children, so adding a
sensor in a later phase cannot change the trajectory of the plant.

TODO(real-hw): ``SimPowerSource`` and ``SimRobot`` are the two adapters to
replace.  Everything above this file consumes only ``PowerSourceBase`` /
``RobotBase`` / ``GroundTruth``-shaped data, and ``GroundTruth`` degrades to
"whatever the real cell can actually tell us" (the estimator never reads the
fields a real cell could not provide — see ``sim/sensors.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np

from weldloop._fastmath import clip
from weldloop.config import WeldConfig, default_config
from weldloop.interfaces import PowerSourceBase, RobotBase
from weldloop.physics.arc import (
    ArcModel,
    ArcSample,
    arc_voltage,
    melting_rate,
    stickout_resistance,
)
from weldloop.physics.melt_pool import (
    DefectFlags,
    MeltPoolModel,
    PoolDerived,
    PoolInputs,
    PoolState,
)
from weldloop.sim.seam import Seam, make_seam

__all__ = [
    "OrnsteinUhlenbeck",
    "TorchCommand",
    "GroundTruth",
    "SimPowerSource",
    "SimRobot",
    "WeldCell",
]


class OrnsteinUhlenbeck:
    """Zero-mean OU process with correlation time ``tau`` and stationary ``std``."""

    def __init__(
        self, tau: float, std: float, rng: np.random.Generator, x0: float = 0.0
    ) -> None:
        self.tau = max(float(tau), 1.0e-6)
        self.std = float(std)
        self._rng = rng
        self.x = float(x0)

    def step(self, dt: float) -> float:
        self.x += (
            -self.x * dt / self.tau
            + self.std * math.sqrt(2.0 * dt / self.tau) * self._rng.standard_normal()
        )
        return self.x


# --------------------------------------------------------------------------
# commands and ground truth
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TorchCommand:
    """Everything a controller can ask the cell to do.

    ``I_set`` and ``arc_len_set`` together define the machine's set voltage
    (that is how a synergic CV machine is programmed); the *actual* current is
    an outcome of self-regulation, not a command.
    """

    I_set: float
    v_wire_set: float
    arc_len_set: float
    v_travel: float
    weave_amp: float = 0.0
    torch_angle: float = 0.0
    ctwd: float = 20.0e-3

    @staticmethod
    def from_config(cfg: WeldConfig) -> "TorchCommand":
        """The nominal (baseline) command."""
        b = cfg.baseline
        return TorchCommand(
            I_set=b.I_set,
            v_wire_set=b.v_wire,
            arc_len_set=cfg.arc.L_arc_ref,
            v_travel=b.v_travel,
            weave_amp=b.weave_amp,
            torch_angle=cfg.robot.torch_angle,
            ctwd=cfg.robot.ctwd_nom,
        )


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Complete state of the cell at one master-clock tick.

    Fields marked *(unobservable)* are the ones a real cell cannot measure;
    they exist so the demo can score the estimator, and nothing downstream of
    ``sim/sensors.py`` is allowed to read them.
    """

    t: float
    s: float
    gap: float  # (unobservable in real time; the profiler measures it ahead)
    offset: float
    pool: PoolState  # (unobservable)
    derived: PoolDerived  # (unobservable)
    defects: DefectFlags  # (unobservable)
    arc: ArcSample
    stickout: float
    arc_length: float
    ctwd: float
    v_travel: float
    weave_offset: float
    weave_amp: float
    v_wire: float
    smoke: float
    torch_force: float
    command: TorchCommand
    done: bool = False


# --------------------------------------------------------------------------
# actuators
# --------------------------------------------------------------------------
class SimPowerSource(PowerSourceBase):
    """Constant-voltage inverter with GMAW self-regulation.

    States: output current ``I`` (fast inner loop, ``tau_I``), wire feed speed
    ``v_wire`` (drive dynamics, ``tau_wire``) and the electrode extension
    ``stickout``.  The extension is the physically interesting one::

        d(stickout)/dt = v_wire - melting_rate(I, stickout)
        L_arc          = CTWD - stickout
        I              = (V_set - V_0 - E_a * L_arc) / R_stickout(stickout)

    Feed faster than the wire melts and the extension grows, the arc shortens,
    the machine pushes more current, the wire melts faster — the loop closes
    itself in ~10 ms.  This *is* what the inverter already does; the ms-level
    inner loop in ``control/inner_loop.py`` only trims its setpoints.

    TODO(real-hw): replace with the machine's fieldbus adapter.  ``measured``
    is the only thing the rest of the code reads.
    """

    def __init__(self, cfg: WeldConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self._rng = rng
        self.ctwd = cfg.robot.ctwd_nom
        #: multiplicative drift on the electrode-extension resistivity
        #: (contact-tip wear).  The estimator does not know about it.
        self.rho_scale = 1.0
        cmd = TorchCommand.from_config(cfg)
        self.I = cmd.I_set
        self.v_wire = cmd.v_wire_set
        self.stickout = cfg.robot.ctwd_nom - cmd.arc_len_set
        #: instantaneous arc length, written by the cell each step (the pool
        #: surface is part of the circuit, so the machine cannot know it a priori)
        self.arc_len_inst = cmd.arc_len_set
        self.V_set = arc_voltage(
            cmd.I_set, cmd.arc_len_set, self.stickout, cfg.arc, cfg.consumable
        )
        self.V = self.V_set
        self._v_wire_set = cmd.v_wire_set

    # -- PowerSourceBase --------------------------------------------------
    def command(self, *, I_set: float, v_wire_set: float, arc_len_set: float) -> None:
        """Program the machine: a synergic (current, arc length) pair sets V."""
        c = self.cfg
        self._v_wire_set = clip(
            v_wire_set, c.power_source.v_wire_min, c.power_source.v_wire_max
        )
        I_set = clip(I_set, c.power_source.I_min, c.power_source.I_max)
        # the set voltage is computed with the NOMINAL stickout, exactly as a
        # machine programmed from a synergic line would
        nominal_stickout = c.robot.ctwd_nom - arc_len_set
        self.V_set = arc_voltage(
            I_set, arc_len_set, nominal_stickout, c.arc, c.consumable
        )

    def step(self, dt: float) -> None:
        """Advance current, wire feed and electrode extension by ``dt``."""
        c = self.cfg
        ps, arc, cons = c.power_source, c.arc, c.consumable

        L_arc = self.arc_len_inst if self.arc_len_inst > ps.L_arc_min else ps.L_arc_min
        R_so = stickout_resistance(self.stickout, cons, self.rho_scale)
        I_target = (self.V_set - arc.V_0 - arc.E_a * L_arc) / max(R_so, 1.0e-6)
        I_target = clip(I_target, ps.I_min, ps.I_max)

        # The current loop is deliberately slower than the pool oscillation, so
        # the machine tracks the mean but not the ripple.
        self.I += (I_target - self.I) * dt / ps.tau_I
        self.v_wire += (self._v_wire_set - self.v_wire) * dt / ps.tau_wire

        # GMAW self-regulation: feed faster than you melt and the extension grows
        d_stickout = self.v_wire - melting_rate(self.I, self.stickout, arc)
        self.stickout = clip(
            self.stickout + d_stickout * dt, ps.stickout_min, self.ctwd - ps.L_arc_min
        )
        self.V = arc.V_0 + arc.E_a * L_arc + R_so * self.I

    @property
    def measured(self) -> dict[str, float]:
        return {
            "I": float(self.I),
            "V": float(self.V),
            "v_wire": float(self.v_wire),
            "V_set": float(self.V_set),
        }

    @property
    def arc_length_geom(self) -> float:
        """Geometric standoff CTWD - stickout [m], before pool depression."""
        L = self.ctwd - self.stickout
        return L if L > self.cfg.power_source.L_arc_min else self.cfg.power_source.L_arc_min


class SimRobot(RobotBase):
    """Torch carrier: travel-speed servo, weave oscillator, height command.

    TODO(real-hw): replace with an EGM / RSI motion-streaming adapter.  The
    controller only ever writes ``command`` and reads ``state``.
    """

    def __init__(self, cfg: WeldConfig) -> None:
        self.cfg = cfg
        cmd = TorchCommand.from_config(cfg)
        self.s = 0.0
        self.v_travel = cmd.v_travel
        self._v_set = cmd.v_travel
        self.weave_amp = cmd.weave_amp
        self._weave_amp_set = cmd.weave_amp
        self.torch_angle = cmd.torch_angle
        self.ctwd = cmd.ctwd
        self.phase = 0.0

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
        # acceleration-limited first-order speed servo
        dv = (self._v_set - self.v_travel) * dt / r.tau_v
        dv = clip(dv, -r.a_travel_max * dt, r.a_travel_max * dt)
        self.v_travel += dv
        # the weave amplitude cannot change instantly either
        self.weave_amp += (self._weave_amp_set - self.weave_amp) * dt / r.tau_v
        self.phase = (self.phase + 2.0 * math.pi * r.weave_freq * dt) % (2.0 * math.pi)
        self.s += self.v_travel * dt

    @property
    def weave_offset(self) -> float:
        return self.weave_amp * math.sin(self.phase)

    @property
    def state(self) -> dict[str, float]:
        return {
            "s": float(self.s),
            "v_travel": float(self.v_travel),
            "weave_amp": float(self.weave_amp),
            "weave_offset": self.weave_offset,
            "ctwd": float(self.ctwd),
            "torch_angle": float(self.torch_angle),
        }


# --------------------------------------------------------------------------
# the cell
# --------------------------------------------------------------------------
@dataclass
class _Streams:
    """Named RNG streams, so that adding a consumer never shifts another."""

    arc: np.random.Generator
    disturbance: np.random.Generator
    sensors: np.random.Generator
    spare: np.random.Generator = field(repr=False, default=None)  # type: ignore[assignment]


class WeldCell:
    """The plant.  One ``step()`` advances the whole cell by ``cfg.sim.dt``."""

    def __init__(
        self,
        cfg: WeldConfig | None = None,
        seam: Seam | None = None,
        seed: int | None = None,
        robot: RobotBase | None = None,
    ) -> None:
        """``robot`` replaces the built-in travel-speed servo.

        This is the seam the abstract base class exists for: passing a
        ``MujocoRobot`` here puts a real articulated arm in the loop and the
        weld physics then sees the TCP the arm actually achieved, tracking
        error and all.  Nothing else in the cell changes.
        """
        self.cfg = cfg or default_config()
        if seed is not None:
            self.cfg.sim.seed = int(seed)
        self.seam = seam if seam is not None else make_seam(self.cfg.seam, self.cfg.sim.seed)
        self._robot_override = robot
        self.reset()

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        cfg = self.cfg
        master = np.random.default_rng(cfg.sim.seed)
        a, d, s, sp = master.spawn(4)
        self.rng = _Streams(arc=a, disturbance=d, sensors=s, spare=sp)

        self.model = MeltPoolModel(cfg)
        self.arc_model = ArcModel(cfg.arc, cfg.consumable, cfg.material, self.rng.arc)
        self.power_source = SimPowerSource(cfg, self.rng.arc)
        self.robot = self._robot_override if self._robot_override is not None else SimRobot(cfg)

        dcfg = cfg.disturbance
        self._ou_rho = OrnsteinUhlenbeck(dcfg.rho_e_tau, dcfg.rho_e_std, self.rng.disturbance)
        self._ou_ctwd = OrnsteinUhlenbeck(dcfg.ctwd_tau, dcfg.ctwd_std, self.rng.disturbance)
        self._ou_smoke = OrnsteinUhlenbeck(dcfg.smoke_tau, dcfg.smoke_std, self.rng.disturbance)

        self.t = 0.0
        self.command_state = TorchCommand.from_config(cfg)
        self.pool = self._ignited_state()
        self._derived = self.model.derived(self.pool, self._pool_inputs(gap=0.0))
        self.done = False

    def _ignited_state(self) -> PoolState:
        """Start from the equilibrium pool for the nominal command at zero gap.

        Starting cold would spend the first several seconds of every run on an
        ignition transient that is not what this demo is about.  The transient
        is available by passing ``cold_start=True`` to :meth:`start_cold`.
        """
        u = self._pool_inputs(gap=float(self.seam.gap_at(0.0)))
        return self.model.steady_state(u)

    def start_cold(self) -> None:
        """Reset the pool to a just-ignited state (for the ignition test)."""
        self.pool = self.model.initial_state()

    # -- plumbing --------------------------------------------------------
    def _pool_inputs(self, gap: float) -> PoolInputs:
        cmd = self.command_state
        return PoolInputs(
            I=self.power_source.I,
            V=self.power_source.V,
            v_travel=self.robot.v_travel,
            v_wire=self.power_source.v_wire,
            gap=gap,
            thickness=self.cfg.joint.thickness,
            weave_amp=self.robot.weave_amp,
        )

    def command(self, cmd: TorchCommand) -> None:
        """Apply a new command to both actuators."""
        self.command_state = cmd
        self.power_source.command(
            I_set=cmd.I_set, v_wire_set=cmd.v_wire_set, arc_len_set=cmd.arc_len_set
        )
        self.robot.command(
            v_travel=cmd.v_travel,
            weave_amp=cmd.weave_amp,
            torch_angle=cmd.torch_angle,
            ctwd=cmd.ctwd,
        )

    # -- the step --------------------------------------------------------
    def step(self) -> GroundTruth:
        """Advance the cell by one simulation timestep and return ground truth."""
        cfg = self.cfg
        dt = cfg.sim.dt

        # 1. motion
        self.robot.step(dt)
        s_pos = min(self.robot.s, self.seam.length)
        gap = float(self.seam.gap_at(s_pos))
        offset = float(self.seam.offset_at(s_pos))

        # 2. slow disturbances the estimator does not know about
        self.power_source.rho_scale = 1.0 + self._ou_rho.step(dt)
        ctwd_true = self.robot.ctwd + self._ou_ctwd.step(dt)
        self.power_source.ctwd = ctwd_true

        # 3. pool-surface oscillation sets the instantaneous arc length.  This
        #    happens BEFORE the machine steps, because the pool surface is part
        #    of the circuit: the inverter reacts to it, it does not command it.
        L_inst, L_mean = self.arc_model.advance(
            dt,
            L_geom=self.power_source.arc_length_geom,
            w=self.pool.w,
            p=self.pool.p,
            superheat=self._derived.superheat,
        )

        # 4. power source self-regulation, then the electrical sample
        self.power_source.arc_len_inst = L_inst
        self.power_source.step(dt)
        arc_sample = self.arc_model.emit(
            dt,
            I=self.power_source.I,
            L_inst=L_inst,
            L_mean=L_mean,
            stickout=self.power_source.stickout,
            superheat=self._derived.superheat,
            rho_scale=self.power_source.rho_scale,
        )

        # 5. melt pool, driven by the INSTANTANEOUS electrical power.  Its
        #    fastest time constant is 20 ms against a 2 ms short circuit, so it
        #    averages the waveform on its own — no separate filtering needed.
        u = PoolInputs(
            I=arc_sample.I,
            V=arc_sample.V,
            v_travel=self.robot.v_travel,
            v_wire=self.power_source.v_wire,
            gap=gap,
            thickness=cfg.joint.thickness,
            weave_amp=self.robot.weave_amp,
        )
        self.pool, self._derived = self.model.step(self.pool, u, dt)
        flags = self.model.defects(self.pool, u)

        # 6. secondary observables
        smoke = max(
            cfg.sensors.smoke_base
            + cfg.sensors.smoke_per_kW * self._derived.q_arc * 1.0e-3
            + self._ou_smoke.step(dt),
            0.0,
        )
        force = cfg.arc.k_arc_force * arc_sample.I**2 + (
            cfg.arc.F_short if arc_sample.in_short else 0.0
        )

        self.t += dt
        self.done = self.robot.s >= self.seam.length or self.t >= cfg.sim.t_max

        return GroundTruth(
            t=self.t,
            s=s_pos,
            gap=gap,
            offset=offset,
            pool=self.pool,
            derived=self._derived,
            defects=flags,
            arc=arc_sample,
            stickout=float(self.power_source.stickout),
            arc_length=float(arc_sample.L_mean),
            ctwd=float(ctwd_true),
            v_travel=float(self.robot.v_travel),
            weave_offset=self.robot.weave_offset,
            weave_amp=float(self.robot.weave_amp),
            v_wire=float(self.power_source.v_wire),
            smoke=float(smoke),
            torch_force=float(force),
            command=self.command_state,
            done=self.done,
        )

    def run(self, max_steps: int | None = None):
        """Yield ``GroundTruth`` until the torch reaches the end of the seam."""
        n = 0
        while not self.done:
            yield self.step()
            n += 1
            if max_steps is not None and n >= max_steps:
                return
