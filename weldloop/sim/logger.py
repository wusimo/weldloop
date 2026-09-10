"""Time-aligned logging onto a single master clock.

This module is written to be read by the customer, not only by the code: the
schema it produces **is the proposed phase-1 real-world capture schema**.  If
the capture rig writes this table, everything downstream in this repository
runs on real data without modification.

Design rules
------------
1. **One master clock.**  Every row is a tick of a single clock (default 5 kHz,
   the power-source rate).  Sensors are binned onto it by their own timestamps,
   not by when the software got round to reading them.
2. **NaN means "did not sample".**  A 30 Hz camera contributes a value on one
   row in 167 and NaN on the rest.  Nothing is forward-filled at write time —
   interpolation is an analysis choice, and baking it into the log destroys the
   evidence of when data actually arrived.
3. **Channels faster than the master clock are binned, not dropped.**  The
   20 kHz microphone contributes an RMS per master tick and a click count per
   master tick.  A real rig logs the raw stream separately; the reducer column
   in the schema says exactly what each master-clock value means.
4. **Ground truth is quarantined.**  Every column that a real cell cannot
   produce is prefixed ``truth_`` and flagged ``real_hw=False``.  The estimator
   and the controllers never read a ``truth_`` column; the scoring code does.

TODO(real-hw): replace the writer with the rig's DAQ, keep ``SCHEMA``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from weldloop.interfaces import SensorSample
from weldloop.sim.cell import GroundTruth

__all__ = ["ColumnSpec", "SCHEMA", "MasterClockLogger", "LogTable", "schema_markdown"]

Reducer = Literal["last", "mean", "rms", "sum", "max"]


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One column of the wide table."""

    name: str
    unit: str
    source: str
    rate_hz: float
    reducer: Reducer
    real_hw: bool
    description: str


#: The capture schema.  Order here is column order in the file.
SCHEMA: tuple[ColumnSpec, ...] = (
    # -- clock ------------------------------------------------------------
    ColumnSpec("t", "s", "master clock", 5000.0, "last", True,
               "master clock time; one row per tick"),
    # -- power source: the primary process sensor -------------------------
    ColumnSpec("ps_V", "V", "power source", 5000.0, "last", True,
               "arc voltage, raw (short-circuit collapses included)"),
    ColumnSpec("ps_I", "A", "power source", 5000.0, "last", True,
               "welding current, raw (short-circuit surges included)"),
    ColumnSpec("ps_short", "-", "power source", 5000.0, "max", True,
               "1 while a short circuit is active"),
    ColumnSpec("ps_v_wire", "m/s", "power source", 5000.0, "last", True,
               "wire feed speed from the drive tacho"),
    ColumnSpec("ps_V_set", "V", "power source", 5000.0, "last", True,
               "machine set voltage (what the CV loop is holding)"),
    # -- geometry ---------------------------------------------------------
    ColumnSpec("prof_gap", "m", "laser profiler", 30.0, "last", True,
               "root gap measured AHEAD of the arc; NaN on spatter dropout"),
    ColumnSpec("prof_offset", "m", "laser profiler", 30.0, "last", True,
               "lateral seam offset ahead of the arc"),
    ColumnSpec("prof_lead_s", "m", "laser profiler", 30.0, "last", True,
               "seam station the profiler was looking at"),
    ColumnSpec("prof_valid", "-", "laser profiler", 30.0, "last", True,
               "0 on dropout; a dropout is never reported as a plausible number"),
    # -- thermal ----------------------------------------------------------
    ColumnSpec("ir_T_peak", "K", "IR camera", 30.0, "last", True,
               "peak apparent pool temperature; smoke-attenuated"),
    ColumnSpec("ir_pool_width", "m", "IR camera", 30.0, "last", True,
               "pool width from the melting isotherm"),
    ColumnSpec("ir_valid", "-", "IR camera", 30.0, "last", True,
               "0 when the frame is rejected (dense plume)"),
    # -- acoustics --------------------------------------------------------
    ColumnSpec("mic_p", "Pa", "arc microphone", 20000.0, "rms", True,
               "RMS over the master tick of the 20 kHz acoustic pressure"),
    ColumnSpec("mic_click", "count", "arc microphone", 20000.0, "sum", True,
               "short-circuit re-ignition clicks in this master tick"),
    # -- mechanical -------------------------------------------------------
    ColumnSpec("force_N", "N", "torch force", 1000.0, "last", True,
               "torch reaction force; dominated by arc force"),
    # -- visible light: present in order to fail --------------------------
    ColumnSpec("rgb_quality", "-", "RGB camera", 30.0, "last", True,
               "image usability = smoke transmission x glare rejection"),
    ColumnSpec("rgb_pool_width", "m", "RGB camera", 30.0, "last", True,
               "pool width from the visible image; noise scales as 1/quality"),
    ColumnSpec("rgb_valid", "-", "RGB camera", 30.0, "last", True,
               "0 when quality is below the usable threshold"),
    # -- commands and machine kinematics ----------------------------------
    ColumnSpec("cmd_I_set", "A", "controller", 5000.0, "last", True,
               "commanded current setpoint"),
    ColumnSpec("cmd_v_wire_set", "m/s", "controller", 5000.0, "last", True,
               "commanded wire feed speed"),
    ColumnSpec("cmd_arc_len_set", "m", "controller", 5000.0, "last", True,
               "commanded arc length"),
    ColumnSpec("cmd_v_travel", "m/s", "controller", 5000.0, "last", True,
               "commanded travel speed"),
    ColumnSpec("cmd_weave_amp", "m", "controller", 5000.0, "last", True,
               "commanded weave half-amplitude"),
    ColumnSpec("rb_s", "m", "robot encoder", 5000.0, "last", True,
               "torch position along the seam"),
    ColumnSpec("rb_v_travel", "m/s", "robot encoder", 5000.0, "last", True,
               "actual travel speed"),
    ColumnSpec("rb_weave_offset", "m", "robot encoder", 5000.0, "last", True,
               "instantaneous lateral weave offset"),
    ColumnSpec("rb_ctwd", "m", "robot", 5000.0, "last", True,
               "commanded contact-tip-to-work distance"),
    # -- ground truth: NOT available on real hardware ---------------------
    ColumnSpec("truth_gap", "m", "simulator", 5000.0, "last", False,
               "true root gap under the arc"),
    ColumnSpec("truth_offset", "m", "simulator", 5000.0, "last", False,
               "true lateral misalignment"),
    ColumnSpec("truth_T_pool", "K", "simulator", 5000.0, "last", False,
               "true mean pool temperature"),
    ColumnSpec("truth_pool_w", "m", "simulator", 5000.0, "last", False,
               "true pool width"),
    ColumnSpec("truth_penetration", "m", "simulator", 5000.0, "last", False,
               "true penetration depth — the quantity being estimated"),
    ColumnSpec("truth_fill", "-", "simulator", 5000.0, "last", False,
               "true gap fill ratio"),
    ColumnSpec("truth_stickout", "m", "simulator", 5000.0, "last", False,
               "true electrode extension"),
    ColumnSpec("truth_arc_len", "m", "simulator", 5000.0, "last", False,
               "true mean arc length including pool depression"),
    ColumnSpec("truth_f_osc", "Hz", "simulator", 5000.0, "last", False,
               "true pool oscillation frequency"),
    ColumnSpec("truth_a_osc", "m", "simulator", 5000.0, "last", False,
               "true pool oscillation amplitude"),
    ColumnSpec("truth_f_sc", "Hz", "simulator", 5000.0, "last", False,
               "true expected short-circuit rate"),
    ColumnSpec("truth_smoke", "-", "simulator", 5000.0, "last", False,
               "true smoke density"),
    ColumnSpec("truth_burn_through", "-", "simulator", 5000.0, "max", False,
               "1 while the burn-through condition holds"),
    ColumnSpec("truth_lack_of_fusion", "-", "simulator", 5000.0, "max", False,
               "1 while any lack-of-fusion condition holds"),
)

_BY_NAME = {c.name: c for c in SCHEMA}

#: columns a real welding cell can actually produce
REAL_HW_COLUMNS: tuple[str, ...] = tuple(c.name for c in SCHEMA if c.real_hw)
#: columns that exist only because this is a simulator
TRUTH_COLUMNS: tuple[str, ...] = tuple(c.name for c in SCHEMA if not c.real_hw)


# --------------------------------------------------------------------------
@dataclass(slots=True)
class LogTable:
    """The finished wide table."""

    data: dict[str, np.ndarray]
    n_rows: int
    f_master: float

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.data.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def real_hw_view(self) -> "LogTable":
        """The same table with every ``truth_`` column removed.

        This is what the estimator and the controllers are allowed to see.
        """
        return LogTable(
            data={k: v for k, v in self.data.items() if not k.startswith("truth_")},
            n_rows=self.n_rows,
            f_master=self.f_master,
        )

    def to_pandas(self, float32: bool = True):
        """As a DataFrame.  Everything but the clock is float32 by default —
        a 5 kHz log is hundreds of thousands of rows and no sensor in the cell
        has seven significant digits."""
        import pandas as pd

        if not float32:
            return pd.DataFrame(self.data)
        return pd.DataFrame(
            {
                k: (v if k == "t" else v.astype(np.float32))
                for k, v in self.data.items()
            }
        )

    def write(self, path: str | Path) -> Path:
        """Write Parquet if pyarrow is present, else gzipped CSV.

        The extension in ``path`` is respected if it is ``.csv``/``.csv.gz``;
        otherwise Parquet is preferred because a 5 kHz log is a few hundred
        thousand rows and CSV of that is neither small nor fast.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        want_csv = path.suffix == ".csv" or path.name.endswith(".csv.gz")
        if not want_csv:
            try:
                import pyarrow  # noqa: F401

                df = self.to_pandas()
                path = path.with_suffix(".parquet")
                df.to_parquet(path, index=False, compression="snappy")
                return path
            except ImportError:
                path = path.with_suffix(".csv.gz")
        df = self.to_pandas()
        df.to_csv(path, index=False, float_format="%.6g")
        return path

    def summary(self) -> str:
        lines = [
            f"rows={self.n_rows}  master={self.f_master:.0f} Hz  "
            f"duration={self.n_rows / self.f_master:.2f} s  cols={len(self.data)}"
        ]
        for name, arr in self.data.items():
            filled = int(np.count_nonzero(~np.isnan(arr)))
            lines.append(
                f"  {name:<22s} {filled:>8d} / {self.n_rows} "
                f"({100.0 * filled / max(self.n_rows, 1):5.1f} %)"
            )
        return "\n".join(lines)


class MasterClockLogger:
    """Bins sensor samples and ground truth onto one master clock.

    The inner loop runs at 5 kHz with ~35 channel writes per tick, so the
    column lookup is resolved once into ``(reducer_code, array)`` pairs and the
    per-tick columns are written through cached array references.  Everything
    here is bookkeeping; the physics is elsewhere.
    """

    _LAST, _MAX, _SUM, _MEAN, _RMS = range(5)
    _CODES = {"last": _LAST, "max": _MAX, "sum": _SUM, "mean": _MEAN, "rms": _RMS}

    #: written once per master tick from ``log_truth``
    _TICK_COLUMNS = (
        "ps_v_wire", "cmd_I_set", "cmd_v_wire_set", "cmd_arc_len_set",
        "cmd_v_travel", "cmd_weave_amp", "rb_s", "rb_v_travel",
        "rb_weave_offset", "rb_ctwd",
    )
    _TRUTH_TICK_COLUMNS = (
        "truth_gap", "truth_offset", "truth_T_pool", "truth_pool_w",
        "truth_penetration", "truth_fill", "truth_stickout", "truth_arc_len",
        "truth_f_osc", "truth_a_osc", "truth_f_sc", "truth_smoke",
        "truth_burn_through", "truth_lack_of_fusion",
    )

    def __init__(
        self, f_master: float, *, include_truth: bool = True, capacity: int = 1 << 16
    ) -> None:
        self.f_master = float(f_master)
        self.include_truth = include_truth
        self._specs = [c for c in SCHEMA if include_truth or c.real_hw]
        self._cap = int(capacity)
        self._n = 0
        #: first row a master tick actually wrote; sensor samples that arrive
        #: before the cell has stepped once are pre-weld and get trimmed
        self._row_lo: int | None = None
        self._data = {
            c.name: np.full(self._cap, np.nan, dtype=np.float64) for c in self._specs
        }
        self._acc = {
            c.name: np.zeros(self._cap, dtype=np.float64)
            for c in self._specs
            if c.reducer in ("mean", "rms")
        }
        self._cnt = {
            c.name: np.zeros(self._cap, dtype=np.int32)
            for c in self._specs
            if c.reducer in ("mean", "rms")
        }
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        self._fast = {
            c.name: (self._CODES[c.reducer], self._data[c.name]) for c in self._specs
        }
        names = self._TICK_COLUMNS + (
            self._TRUTH_TICK_COLUMNS if self.include_truth else ()
        )
        self._tick_arrays = tuple(self._data[n] for n in names if n in self._data)
        self._tick_names = tuple(n for n in names if n in self._data)

    # -- capacity --------------------------------------------------------
    def _grow(self, needed: int) -> None:
        cap = self._cap
        while cap <= needed:
            cap *= 2
        for name, arr in self._data.items():
            new = np.full(cap, np.nan, dtype=np.float64)
            new[: self._cap] = arr
            self._data[name] = new
        for store, fill in ((self._acc, 0.0), (self._cnt, 0)):
            for name, arr in store.items():
                new = np.full(cap, fill, dtype=arr.dtype)
                new[: self._cap] = arr
                store[name] = new
        self._cap = cap
        self._rebuild_cache()

    def _row(self, t: float) -> int:
        row = int(t * self.f_master + 0.5)
        if row >= self._cap:
            self._grow(row)
        if row >= self._n:
            self._n = row + 1
        return row

    # -- writing ---------------------------------------------------------
    def _put(self, name: str, row: int, value: float) -> None:
        ent = self._fast.get(name)
        if ent is None or value != value:  # unknown channel, or NaN
            return
        code, arr = ent
        if code == 0:  # last
            arr[row] = value
        elif code == 1:  # max
            cur = arr[row]
            if cur != cur or value > cur:
                arr[row] = value
        elif code == 2:  # sum
            cur = arr[row]
            arr[row] = value if cur != cur else cur + value
        else:  # mean / rms
            self._acc[name][row] += value * value if code == 4 else value
            self._cnt[name][row] += 1

    def log_samples(self, samples: Iterable[SensorSample]) -> None:
        """Bin a block of sensor samples by their own timestamps."""
        put = self._put
        row_of = self._row
        for smp in samples:
            row = row_of(smp.t)
            for name, value in smp.channels.items():
                put(name, row, value)

    def log_truth(self, truth: GroundTruth) -> None:
        """Record the master-clock row: commands, kinematics and ground truth."""
        row = self._row(truth.t)
        if self._row_lo is None:
            self._row_lo = row
        cmd = truth.command
        values = [
            truth.v_wire, cmd.I_set, cmd.v_wire_set, cmd.arc_len_set,
            cmd.v_travel, cmd.weave_amp, truth.s, truth.v_travel,
            truth.weave_offset, cmd.ctwd,
        ]
        if self.include_truth:
            d = truth.defects
            pool = truth.pool
            values += [
                truth.gap, truth.offset, pool.T_pool, pool.w, pool.p, pool.f,
                truth.stickout, truth.arc_length, truth.arc.f_osc,
                truth.arc.a_osc, truth.arc.f_sc, truth.smoke,
                float(d.burn_through), float(d.lack_of_fusion),
            ]
        for arr, value in zip(self._tick_arrays, values):
            arr[row] = value

    def log_V_set(self, V_set: float, t: float) -> None:
        self._put("ps_V_set", self._row(t), V_set)

    # -- finish ----------------------------------------------------------
    def finish(self) -> LogTable:
        """Resolve the aggregating reducers and trim to the used length.

        Leading rows before the plant's first tick are dropped, so every
        per-tick column (including the ground truth) is complete on every row
        that survives.  A partially-populated first row is a trap for anyone
        who later writes ``table["truth_gap"].max()``.
        """
        lo = self._row_lo or 0
        out: dict[str, np.ndarray] = {}
        n_rows = max(self._n - lo, 0)
        for spec in self._specs:
            if spec.name == "t":
                # The master clock is DEFINED by the row index, so it is always
                # complete even on a tick where nothing sampled.
                out["t"] = (np.arange(n_rows, dtype=np.float64) + lo) / self.f_master
                continue
            arr = self._data[spec.name][lo : self._n]
            if spec.reducer in ("mean", "rms"):
                cnt = self._cnt[spec.name][lo : self._n]
                acc = self._acc[spec.name][lo : self._n]
                with np.errstate(invalid="ignore", divide="ignore"):
                    val = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
                    if spec.reducer == "rms":
                        val = np.sqrt(val)
                arr = val
            out[spec.name] = arr
        return LogTable(data=out, n_rows=n_rows, f_master=self.f_master)


# --------------------------------------------------------------------------
def schema_markdown(include_truth: bool = True) -> str:
    """The schema as a Markdown table, for pasting into the README."""
    head = (
        "| column | unit | source | rate [Hz] | 主时钟聚合 | 真实产线可得 | 说明 |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for c in SCHEMA:
        if not include_truth and not c.real_hw:
            continue
        rows.append(
            f"| `{c.name}` | {c.unit} | {c.source} | {c.rate_hz:g} | "
            f"{c.reducer} | {'✅' if c.real_hw else '❌ 仅仿真'} | {c.description} |"
        )
    return head + "\n".join(rows)
