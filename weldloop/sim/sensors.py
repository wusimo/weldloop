"""Synthetic sensor suite.

Six sensors, six different rates, one master clock.  Each models the failure
mode that actually matters for it in a welding cell:

======================  ======  ==================================================
sensor                  rate    what it gets wrong
======================  ======  ==================================================
``PowerSourceSensor``   5 kHz   Gaussian noise + ADC quantisation.  Never blinded.
                                This is the point: it is the only sensor that
                                keeps working through smoke, glare and spatter.
``SeamProfiler``        30 Hz   spatter dropouts; measures AHEAD of the arc, so
                                it gives the controller preview, not feedback
``IRCamera``            30 Hz   smoke attenuates the radiance; the camera is
                                calibrated at a nominal smoke level, so what is
                                left is variance, plus hard rejection in a plume
``ArcMic``              20 kHz  broadband + a tone at the pool oscillation +
                                impulsive clicks on short-circuit re-ignition
``TorchForce``          1 kHz   noisy, and dominated by arc force, not the pool
``RGBCamera``           30 Hz   included ONLY to demonstrate that it fails:
                                smoke and arc glare drive its quality to ~0 and
                                its pool-width reading to nonsense
======================  ======  ==================================================

Rule the demo must not break: **RGB is never used as a process sensor.**  It is
in the suite so the RGB-only row of the Phase 3 RMSE table can show what
happens if you try.

TODO(real-hw): every class here is one adapter.  A real adapter subclasses
``SensorBase``, ignores the ``GroundTruth`` argument, reads its device, and
emits ``SensorSample`` on the same columns.  Nothing downstream changes.
"""

from __future__ import annotations

import math

import numpy as np

from weldloop.config import SensorConfig, WeldConfig
from weldloop.interfaces import SensorBase, SensorSample
from weldloop.sim.cell import GroundTruth
from weldloop.sim.seam import Seam

__all__ = [
    "PowerSourceSensor",
    "SeamProfiler",
    "IRCamera",
    "ArcMic",
    "TorchForce",
    "RGBCamera",
    "SensorSuite",
]


def _quantise(x: float, lsb: float) -> float:
    return round(x / lsb) * lsb if lsb > 0.0 else x


# --------------------------------------------------------------------------
class PowerSourceSensor(SensorBase):
    """Arc voltage and current at 5 kHz — the primary process sensor.

    Emits the *raw* waveform: the short-circuit collapses are in it, and so is
    the pool-oscillation ripple.  Feature extraction is somebody else's job
    (``estimation/features.py``); a sensor that pre-averages its own signal
    would throw away exactly the information this proposal is about.
    """

    name = "ps"
    columns = ("ps_V", "ps_I", "ps_short")

    def __init__(self, cfg: SensorConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.f_power, rng)
        self.cfg = cfg

    def sample(self, t: float, truth: GroundTruth) -> SensorSample:
        c = self.cfg
        V = truth.arc.V + c.noise_V * self._rng.standard_normal()
        I = truth.arc.I + c.noise_I * self._rng.standard_normal()
        return SensorSample(
            t=t,
            channels={
                "ps_V": _quantise(V, c.lsb_V),
                "ps_I": _quantise(I, c.lsb_I),
                "ps_short": float(truth.arc.in_short),
            },
        )


class SeamProfiler(SensorBase):
    """Laser-line seam profiler at 30 Hz, mounted ahead of the torch.

    Reports the root gap and lateral offset at ``profiler_lead`` metres *ahead*
    of the arc, which is what a real leading-scanner installation gives you:
    preview, arriving ``lead / travel_speed`` seconds before it matters.  The
    controller may use it as feed-forward; the estimator uses it to pin down
    the gap input of the process model.

    Spatter on the window causes frame dropouts.  A dropout is reported as
    ``valid=False`` with NaN channels — never as a plausible-looking wrong
    number, which is the failure mode that actually hurts.
    """

    name = "prof"
    columns = ("prof_gap", "prof_offset", "prof_lead_s", "prof_valid")

    def __init__(self, cfg: SensorConfig, seam: Seam, rng: np.random.Generator) -> None:
        super().__init__(cfg.f_profiler, rng)
        self.cfg = cfg
        self.seam = seam

    def sample(self, t: float, truth: GroundTruth) -> SensorSample:
        c = self.cfg
        s_look = min(truth.s + c.profiler_lead, self.seam.length)
        if self._rng.random() < c.profiler_dropout:
            return SensorSample(
                t=t,
                channels={
                    "prof_gap": math.nan,
                    "prof_offset": math.nan,
                    "prof_lead_s": s_look,
                    "prof_valid": 0.0,
                },
                valid=False,
            )
        gap = float(self.seam.gap_at(s_look)) + c.noise_gap * self._rng.standard_normal()
        off = (
            float(self.seam.offset_at(s_look))
            + c.noise_offset * self._rng.standard_normal()
        )
        return SensorSample(
            t=t,
            channels={
                "prof_gap": max(gap, 0.0),
                "prof_offset": off,
                "prof_lead_s": s_look,
                "prof_valid": 1.0,
            },
        )


class IRCamera(SensorBase):
    """Thermal camera at 30 Hz: peak pool temperature and isotherm width.

    Smoke attenuates the radiance the camera collects.  A real camera is
    calibrated against a nominal weld, so the *mean* attenuation is absorbed
    into the calibration and what survives is the fluctuation::

        T_meas = T_0 + (T_pool - T_0) * exp(-tau * (smoke - smoke_ref)) + noise

    That is the honest version: smoke costs you variance, not a constant bias.
    In a dense plume (``smoke > ir_blind_smoke``) the frame is rejected
    outright.  LWIR penetrates weld fume far better than visible light — which
    is exactly why the IR row of the RMSE table beats the RGB row.
    """

    name = "ir"
    columns = ("ir_T_peak", "ir_pool_width", "ir_valid")

    def __init__(self, cfg: SensorConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.f_ir, rng)
        self.cfg = cfg

    def sample(self, t: float, truth: GroundTruth) -> SensorSample:
        c = self.cfg
        if truth.smoke > c.ir_blind_smoke:
            return SensorSample(
                t=t,
                channels={
                    "ir_T_peak": math.nan,
                    "ir_pool_width": math.nan,
                    "ir_valid": 0.0,
                },
                valid=False,
            )
        atten = math.exp(-c.ir_smoke_tau * (truth.smoke - c.smoke_ref))
        T_amb = 300.0
        T = T_amb + (truth.pool.T_pool - T_amb) * atten
        T += c.noise_T * self._rng.standard_normal()
        w = truth.pool.w * atten + c.noise_w * self._rng.standard_normal()
        return SensorSample(
            t=t,
            channels={
                "ir_T_peak": T,
                "ir_pool_width": max(w, 0.0),
                "ir_valid": 1.0,
            },
        )


class ArcMic(SensorBase):
    """Arc microphone at 20 kHz — four samples per 5 kHz simulation step.

    The synthesised pressure signal has three parts:

    * broadband hiss whose level scales with arc power,
    * a tone at the pool-oscillation frequency (the pool is a loudspeaker for
      its own surface motion),
    * an impulsive click each time the arc re-ignites after a short circuit.

    The logger bins ``mic_p`` to the master clock as an RMS and ``mic_click``
    as a count, because storing a raw 20 kHz channel in a wide table would
    quadruple the file for no analytical gain.  A real rig logs the raw stream
    to its own file and the block features to the master table — the schema
    here is written to match that.
    """

    name = "mic"
    columns = ("mic_p", "mic_click")

    def __init__(self, cfg: SensorConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.f_mic, rng)
        self.cfg = cfg
        self._phase = 0.0
        self._was_short = False
        self._click_left = 0

    def sample(self, t: float, truth: GroundTruth) -> SensorSample:
        c = self.cfg
        dt = self._period
        self._phase = (
            self._phase + 2.0 * math.pi * truth.arc.f_osc * dt
        ) % (2.0 * math.pi)

        level = c.mic_broadband_per_kW * truth.derived.q_arc * 1.0e-3
        p = level * self._rng.standard_normal()
        p += c.f_mic_tone * truth.arc.a_osc / 0.3e-3 * math.sin(self._phase)
        p += c.noise_mic * self._rng.standard_normal()

        # a click fires on re-ignition (arc restrikes), not on contact
        click = 0.0
        if self._was_short and not truth.arc.in_short:
            self._click_left = 3
        self._was_short = truth.arc.in_short
        if self._click_left > 0:
            self._click_left -= 1
            amp = c.mic_click * (0.5 ** (2 - self._click_left))
            p += amp * (1.0 if self._click_left == 2 else -0.6)
            click = 1.0 if self._click_left == 2 else 0.0

        return SensorSample(t=t, channels={"mic_p": p, "mic_click": click})


class TorchForce(SensorBase):
    """Torch-mounted force sensor at 1 kHz.

    Dominated by arc force (``k * I^2``) plus the impulse of wire contact
    during a short circuit.  Included because it is cheap and it corroborates
    the short-circuit statistics, not because it sees the pool — it does not.
    """

    name = "force"
    columns = ("force_N",)

    def __init__(self, cfg: SensorConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.f_force, rng)
        self.cfg = cfg

    def sample(self, t: float, truth: GroundTruth) -> SensorSample:
        F = truth.torch_force + self.cfg.noise_force * self._rng.standard_normal()
        return SensorSample(t=t, channels={"force_N": F})


class RGBCamera(SensorBase):
    """Visible-light camera at 30 Hz.  Present to fail, on purpose.

    Two independent things destroy it during welding:

    * **smoke**, with a far larger attenuation coefficient than the IR band
      (``rgb_smoke_tau`` vs ``ir_smoke_tau``);
    * **arc glare**, which blooms the sensor at exactly the moment there is
      something to look at.

    ``rgb_quality`` is the product of the two.  The pool-width reading is
    emitted with noise scaled by ``1 / quality``, so it degrades continuously
    rather than snapping to invalid — which is the honest failure: the number
    keeps arriving, it just stops meaning anything.  Below
    ``rgb_quality_min`` the frame is marked invalid.
    """

    name = "rgb"
    columns = ("rgb_quality", "rgb_pool_width", "rgb_valid")

    def __init__(self, cfg: SensorConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.f_rgb, rng)
        self.cfg = cfg

    def quality(self, truth: GroundTruth) -> float:
        c = self.cfg
        vis = math.exp(-c.rgb_smoke_tau * truth.smoke)
        glare = 1.0 / (1.0 + (truth.arc.I / c.rgb_glare_I) ** 2)
        return float(vis * glare)

    def sample(self, t: float, truth: GroundTruth) -> SensorSample:
        c = self.cfg
        q = self.quality(truth)
        if q < c.rgb_quality_min:
            return SensorSample(
                t=t,
                channels={
                    "rgb_quality": q,
                    "rgb_pool_width": math.nan,
                    "rgb_valid": 0.0,
                },
                valid=False,
            )
        sigma = c.noise_rgb_w / max(q, 1.0e-3)
        w = truth.pool.w + sigma * self._rng.standard_normal()
        return SensorSample(
            t=t,
            channels={
                "rgb_quality": q,
                "rgb_pool_width": max(w, 0.0),
                "rgb_valid": 1.0,
            },
        )


# --------------------------------------------------------------------------
class SensorSuite:
    """All six sensors on one master clock, each with its own RNG stream.

    Streams are spawned per sensor so that enabling or disabling one sensor
    cannot change what any other sensor produces — without that, the Phase 3
    ablation table ("V/I only" vs "V/I + profiler") would be comparing
    different noise realisations, not different sensor sets.
    """

    def __init__(
        self, cfg: WeldConfig, seam: Seam, rng: np.random.Generator
    ) -> None:
        streams = rng.spawn(6)
        sc = cfg.sensors
        self.power = PowerSourceSensor(sc, streams[0])
        self.profiler = SeamProfiler(sc, seam, streams[1])
        self.ir = IRCamera(sc, streams[2])
        self.mic = ArcMic(sc, streams[3])
        self.force = TorchForce(sc, streams[4])
        self.rgb = RGBCamera(sc, streams[5])
        self.sensors: tuple[SensorBase, ...] = (
            self.power, self.profiler, self.ir, self.mic, self.force, self.rgb
        )

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(col for s in self.sensors for col in s.columns)

    def poll(self, t: float, truth: GroundTruth) -> list[SensorSample]:
        """Every sample from every sensor that has arrived by master time ``t``."""
        out: list[SensorSample] = []
        for s in self.sensors:
            out.extend(s.poll(t, truth))
        return out

    def reset(self) -> None:
        for s in self.sensors:
            s.reset()
