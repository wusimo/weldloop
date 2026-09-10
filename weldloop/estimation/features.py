"""Feature extraction from the 5 kHz V/I stream — "the power source as a sensor".

What this module is for
-----------------------
The mean voltage and mean current of a constant-voltage GMAW arc say almost
nothing about penetration: the wire feed pins the current through mass
conservation and the machine pins the voltage.  Anyone who has tried to
regress penetration on mean current knows this.  What *does* carry information
is the structure of the waveform:

``f_ripple``
    The pool surface oscillates; that modulates the arc length and therefore
    the voltage.  The oscillation frequency follows a surface-tension scaling
    in the pool radius, so **the ripple frequency reports pool width**.

``a_ripple``
    The amplitude of the same modulation grows with pool depth and superheat,
    so **the ripple amplitude reports penetration**.  It is expressed here in
    metres of equivalent arc length (``amplitude_volts / E_a``) so that it can
    be compared directly with the physics.

``f_sc``, ``sc_duty``
    A short circuit happens when the oscillating surface reaches the wire tip,
    so the short-circuit rate is a third, independent view of the same state.
    Weak on its own at this current — the arc is globular, not short-arc — but
    it costs nothing and it is uncorrelated with the ripple estimate.

``L_arc_est``
    Arc length inverted from the static characteristic using the *nominal*
    stickout.  Contact-tip wear and stickout drift bias it, which is precisely
    why the geometric and thermal sensors still earn their place.

Preprocessing that matters: the short-circuit periods are **blanked** (linearly
interpolated across) before the spectrum is taken.  A 20 V collapse would
otherwise bury a 0.4 V ripple, and the spectrum would just show the
short-circuit repetition rate.  ``tests/test_ekf.py`` checks that skipping this
step destroys the feature.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from weldloop.config import TransferMode, WeldConfig
from weldloop.physics.arc import arc_length_from_voltage, classify_transfer_mode

__all__ = [
    "PowerFeatures",
    "blank_short_circuits",
    "ripple_peak",
    "extract_features",
    "FeatureStream",
    "GapTracker",
]


@dataclass(frozen=True, slots=True)
class PowerFeatures:
    """Features of one V/I window.  SI units."""

    t: float
    V_mean: float
    I_mean: float
    V_std: float
    I_std: float
    P_mean: float
    P_std: float
    f_sc: float
    sc_duty: float
    L_arc_est: float
    f_ripple: float
    a_ripple: float
    ripple_snr: float
    mode: TransferMode
    valid: bool

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["valid"] = float(self.valid)
        return d


# --------------------------------------------------------------------------
def blank_short_circuits(V: np.ndarray, short: np.ndarray) -> np.ndarray:
    """Linearly interpolate the arc voltage across short-circuit periods.

    Returns a copy.  If everything is shorted the input is returned unchanged
    (the caller will mark the window invalid anyway).
    """
    shorted = short > 0.5
    if not shorted.any():
        return V
    good = ~shorted
    if good.sum() < 2:
        return V
    out = V.copy()
    idx = np.arange(len(V))
    out[shorted] = np.interp(idx[shorted], idx[good], V[good])
    return out


def ripple_peak(
    x: np.ndarray, dt: float, f_lo: float, f_hi: float, pad: int = 4
) -> tuple[float, float, float]:
    """Dominant sinusoid in ``[f_lo, f_hi]``: (frequency, amplitude, SNR).

    Zero-padded FFT plus parabolic interpolation of the peak, so the frequency
    resolution is not limited to ``1 / window`` — which matters because the
    window has to stay short enough (200 ms) to track a moving pool.

    Amplitude is the physical amplitude of the sinusoid, corrected for the
    coherent gain of the Hann window (0.5).
    """
    n = len(x)
    if n < 16:
        return math.nan, math.nan, 0.0
    x = x - x.mean()
    # remove a linear trend so a drifting arc length does not leak into the band
    t = np.arange(n, dtype=float)
    slope = float(np.polyfit(t, x, 1)[0])
    x = x - slope * (t - t.mean())

    win = np.hanning(n)
    nfft = int(2 ** math.ceil(math.log2(n * pad)))
    spec = np.abs(np.fft.rfft(x * win, n=nfft))
    freqs = np.fft.rfftfreq(nfft, dt)

    band = (freqs >= f_lo) & (freqs <= f_hi)
    if band.sum() < 3:
        return math.nan, math.nan, 0.0
    idx_band = np.flatnonzero(band)
    k_rel = int(np.argmax(spec[band]))
    k = idx_band[k_rel]

    # parabolic interpolation on the log spectrum
    f_peak = float(freqs[k])
    if 0 < k < len(spec) - 1:
        a, b, c = (math.log(max(spec[k + d], 1e-30)) for d in (-1, 0, 1))
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            delta = 0.5 * (a - c) / denom
            f_peak = float(freqs[k] + delta * (freqs[1] - freqs[0]))

    amp = float(2.0 * spec[k] / (n * 0.5))  # 0.5 = Hann coherent gain
    med = float(np.median(spec[band]))
    snr = float(spec[k] / med) if med > 0 else 0.0
    return f_peak, amp, snr


def extract_features(
    t: float,
    V: np.ndarray,
    I: np.ndarray,
    short: np.ndarray,
    cfg: WeldConfig,
) -> PowerFeatures:
    """All V/I features of one window.  ``t`` stamps the window's right edge."""
    dt = 1.0 / cfg.sensors.f_master
    n = len(V)
    arcing = short <= 0.5
    n_arc = int(arcing.sum())
    valid = n_arc >= max(16, int(0.3 * n))

    if not valid:
        nan = math.nan
        return PowerFeatures(
            t=t, V_mean=nan, I_mean=nan, V_std=nan, I_std=nan, P_mean=nan,
            P_std=nan, f_sc=nan, sc_duty=1.0, L_arc_est=nan, f_ripple=nan,
            a_ripple=nan, ripple_snr=0.0, mode=TransferMode.SHORT_CIRCUIT,
            valid=False,
        )

    V_arc, I_arc = V[arcing], I[arcing]
    P = V * I
    duration = n * dt
    # a short circuit is one contiguous run; count its rising edges
    edges = int(np.count_nonzero(np.diff((short > 0.5).astype(np.int8)) > 0))

    Vb = blank_short_circuits(V, short)
    f_rip, amp_V, snr = ripple_peak(
        Vb, dt, cfg.estimator_band[0], cfg.estimator_band[1]
    )
    a_rip = amp_V / cfg.arc.E_a if np.isfinite(amp_V) else math.nan

    I_mean = float(I_arc.mean())
    V_mean = float(V_arc.mean())
    L_est = arc_length_from_voltage(
        V_mean, I_mean, cfg.arc.stickout, cfg.arc, cfg.consumable
    )

    return PowerFeatures(
        t=t,
        V_mean=V_mean,
        I_mean=I_mean,
        V_std=float(V_arc.std()),
        I_std=float(I_arc.std()),
        P_mean=float(P.mean()),
        P_std=float(P.std()),
        f_sc=edges / duration,
        sc_duty=float(np.mean(short > 0.5)),
        L_arc_est=L_est,
        f_ripple=f_rip,
        a_ripple=a_rip,
        ripple_snr=snr,
        mode=classify_transfer_mode(I_mean, cfg.arc),
        valid=True,
    )


# --------------------------------------------------------------------------
class FeatureStream:
    """Ring buffer over the 5 kHz V/I stream, emitting features at a fixed rate.

    Used identically offline (fed from a log) and online (fed from the
    ``Observation``), so the Phase 3 numbers and the Phase 4 controller cannot
    diverge.
    """

    def __init__(self, cfg: WeldConfig) -> None:
        self.cfg = cfg
        self._n = int(round(cfg.estimator_window * cfg.sensors.f_master))
        self._V = np.zeros(self._n)
        self._I = np.zeros(self._n)
        self._s = np.zeros(self._n)
        self._i = 0
        self._filled = 0
        self._period = cfg.ekf.dt
        self._t_next = cfg.estimator_window

    def push(self, V: float, I: float, short: float) -> None:
        i = self._i
        self._V[i] = V
        self._I[i] = I
        self._s[i] = short
        self._i = (i + 1) % self._n
        if self._filled < self._n:
            self._filled += 1

    def due(self, t: float) -> bool:
        return self._filled >= self._n and t + 1e-12 >= self._t_next

    def extract(self, t: float) -> PowerFeatures:
        """Features of the current window; advances the emission schedule."""
        self._t_next += self._period
        order = np.arange(self._i, self._i + self._n) % self._n
        return extract_features(t, self._V[order], self._I[order], self._s[order], self.cfg)


class GapTracker:
    """Turns the look-ahead profiler into the gap *under the arc*.

    The scanner reports the gap at ``s + lead``; the process model needs it at
    ``s``.  So readings are stored against the seam station they describe and
    read back when the torch gets there.  Dropouts simply leave holes that the
    interpolation bridges — which is the right behaviour, because a gap profile
    is smooth on the scale of a dropout.

    Before the first reading arrives, and beyond the last one, it falls back to
    ``default_gap``.  That fallback is what the "V/I only" ablation runs on.
    """

    def __init__(self, default_gap: float = 0.0, capacity: int = 4096) -> None:
        self.default_gap = float(default_gap)
        self._s = np.empty(capacity)
        self._g = np.empty(capacity)
        self._n = 0

    def push(self, s_measured: float, gap: float) -> None:
        if not np.isfinite(s_measured) or not np.isfinite(gap):
            return
        if self._n and s_measured <= self._s[self._n - 1]:
            return  # out of order or stationary; keep the monotone series
        if self._n == len(self._s):
            self._s = np.concatenate([self._s, np.empty_like(self._s)])
            self._g = np.concatenate([self._g, np.empty_like(self._g)])
        self._s[self._n] = s_measured
        self._g[self._n] = gap
        self._n += 1

    def gap_at(self, s: float) -> float:
        if self._n == 0:
            return self.default_gap
        if s < self._s[0]:
            return self.default_gap
        return float(np.interp(s, self._s[: self._n], self._g[: self._n]))

    @property
    def n_readings(self) -> int:
        return self._n
