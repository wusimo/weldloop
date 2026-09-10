"""Optional learned residual on the melt-pool process model.

Why there is something to learn
-------------------------------
The estimator does **not** run the simulator's model: it runs the one from
:func:`config.estimator_config`, whose lumped coefficients are deliberately
off.  That mismatch stands in for the parameter error you are left with after
identifying a reduced-order model from a finite amount of real weld data, and
it produces a systematic penetration bias of a few tenths of a millimetre.

The residual network learns exactly that leftover::

    residual = x_true(t + dt) - f_model(x_true(t), u(t), dt)

so it corrects the *dynamics*, not the measurements.  It is small on purpose —
two hidden layers — because the physics is already carrying the structure and a
big network here would just memorise the seam.

Graceful degradation
--------------------
``torch`` is an optional dependency.  With no torch, no checkpoint, or a
checkpoint that fails to load, :func:`load_residual` returns ``None`` and the
EKF runs pure physics.  Every number in the README is reported both with and
without it, so the contribution of the learned part is always visible and
never load-bearing.

TODO(real-hw): retrain on real captures.  The training target is built from
``truth_*`` columns here; on a real cell the target comes from destructive
macro-sections of test coupons (penetration and width measured on the cut
face), interpolated along the seam.  That is the phase-1 experiment this
residual is designed to consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

__all__ = [
    "ResidualModel",
    "ResidualDataset",
    "build_dataset",
    "MLPResidual",
    "load_residual",
    "torch_available",
]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class ResidualModel(Protocol):
    """What ``PoolEKF`` needs from a residual."""

    def correction(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray: ...


# --------------------------------------------------------------------------
@dataclass(slots=True)
class ResidualDataset:
    """Training pairs: model inputs, and the increment the model got wrong."""

    X: np.ndarray  # (N, 11) = state (4) + inputs (7)
    Y: np.ndarray  # (N, 4)  = x_true_next - f_model(x_true, u)
    dt: float

    def __len__(self) -> int:
        return len(self.X)

    def split(self, frac: float = 0.2, seed: int = 0):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.X))
        cut = int(len(idx) * (1.0 - frac))
        tr, va = idx[:cut], idx[cut:]
        return (
            ResidualDataset(self.X[tr], self.Y[tr], self.dt),
            ResidualDataset(self.X[va], self.Y[va], self.dt),
        )


def build_dataset(cfg, table, n_sub: int = 4) -> ResidualDataset:
    """Extract (state, input) -> model-error pairs from one logged weld.

    Uses ``truth_*`` columns, which only exist in simulation; see the module
    docstring for what replaces them on real hardware.
    """
    from weldloop.config import estimator_config
    from weldloop.physics.melt_pool import MeltPoolModel

    model = MeltPoolModel(estimator_config(cfg))
    dt = cfg.ekf.dt
    step = int(round(dt * table.f_master))
    rows = np.arange(0, table.n_rows - step, step)

    state_cols = ("truth_T_pool", "truth_pool_w", "truth_penetration", "truth_fill")
    if any(c not in table.columns for c in state_cols):
        raise KeyError("build_dataset needs a log with truth_ columns")

    Xs, Ys = [], []
    h = dt / n_sub
    for r in rows:
        x = np.array([table[c][r] for c in state_cols], dtype=float)
        x_next = np.array([table[c][r + step] for c in state_cols], dtype=float)
        u = np.array(
            [
                table["ps_I"][r],
                table["ps_V"][r],
                table["rb_v_travel"][r],
                table["ps_v_wire"][r],
                table["truth_gap"][r],
                cfg.joint.thickness,
                table["cmd_weave_amp"][r],
            ],
            dtype=float,
        )
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(u)) and np.all(np.isfinite(x_next))):
            continue
        pred = x.copy()
        for _ in range(n_sub):
            pred = model.step_array(pred, u, h)
        Xs.append(np.concatenate([x, u]))
        Ys.append(x_next - pred)
    return ResidualDataset(np.asarray(Xs), np.asarray(Ys), dt)


# --------------------------------------------------------------------------
class MLPResidual:
    """A two-hidden-layer MLP correcting the one-step state increment.

    Inputs and outputs are standardised with statistics stored in the
    checkpoint, so the object is self-contained.  The correction is clipped to
    ``clip_sigma`` training standard deviations: a residual network that is
    allowed to make an unbounded correction can walk an EKF off a cliff on an
    operating point it never saw, and the physics must stay in charge.
    """

    def __init__(self, net, x_mean, x_std, y_mean, y_std, dt: float, clip_sigma: float = 3.0):
        self._net = net
        self._x_mean = np.asarray(x_mean, dtype=float)
        self._x_std = np.asarray(x_std, dtype=float)
        self._y_mean = np.asarray(y_mean, dtype=float)
        self._y_std = np.asarray(y_std, dtype=float)
        self.dt = float(dt)
        self.clip = clip_sigma * self._y_std

    @staticmethod
    def build(n_in: int = 11, n_out: int = 4, hidden: int = 64):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_out),
        )

    def correction(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        import torch

        z = (np.concatenate([x, u]) - self._x_mean) / self._x_std
        with torch.no_grad():
            y = self._net(torch.as_tensor(z, dtype=torch.float32)).numpy()
        out = y * self._y_std + self._y_mean
        out = np.clip(out, -self.clip, self.clip)
        # the network was trained at one dt; scale linearly for any other
        return out * (dt / self.dt)

    def save(self, path: str | Path) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "x_mean": self._x_mean, "x_std": self._x_std,
                "y_mean": self._y_mean, "y_std": self._y_std,
                "dt": self.dt,
            },
            path,
        )
        return path


def load_residual(path: str | Path, *, quiet: bool = False) -> ResidualModel | None:
    """Load a trained residual, or return ``None`` and say why.

    Never raises.  The EKF is expected to work without it.
    """
    path = Path(path)
    if not torch_available():
        if not quiet:
            print("[residual] torch not installed - running pure physics")
        return None
    if not path.exists():
        if not quiet:
            print(f"[residual] no checkpoint at {path} - running pure physics")
        return None
    try:
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = MLPResidual.build()
        net.load_state_dict(ck["state_dict"])
        net.eval()
        return MLPResidual(
            net, ck["x_mean"], ck["x_std"], ck["y_mean"], ck["y_std"], ck["dt"]
        )
    except Exception as exc:  # pragma: no cover - defensive
        if not quiet:
            print(f"[residual] failed to load {path}: {exc} - running pure physics")
        return None


def train(
    dataset: ResidualDataset,
    *,
    epochs: int = 400,
    hidden: int = 64,
    lr: float = 3.0e-3,
    batch: int = 512,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[MLPResidual, dict[str, float]]:
    """Train the residual.  Requires torch; raises ImportError otherwise."""
    import torch

    torch.manual_seed(seed)
    tr, va = dataset.split(0.2, seed=seed)

    x_mean, x_std = tr.X.mean(0), tr.X.std(0) + 1e-12
    y_mean, y_std = tr.Y.mean(0), tr.Y.std(0) + 1e-12

    Xtr = torch.as_tensor((tr.X - x_mean) / x_std, dtype=torch.float32)
    Ytr = torch.as_tensor((tr.Y - y_mean) / y_std, dtype=torch.float32)
    Xva = torch.as_tensor((va.X - x_mean) / x_std, dtype=torch.float32)
    Yva = torch.as_tensor((va.Y - y_mean) / y_std, dtype=torch.float32)

    net = MLPResidual.build(Xtr.shape[1], Ytr.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.SmoothL1Loss()
    n = len(Xtr)
    best = math.inf
    best_state = {k: v.clone() for k, v in net.state_dict().items()}

    for ep in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(net(Xva), Yva))
        if val < best:
            best = val
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  val {val:.5f}")

    net.load_state_dict(best_state)
    net.eval()
    model = MLPResidual(net, x_mean, x_std, y_mean, y_std, dataset.dt)

    with torch.no_grad():
        pred = net(Xva).numpy() * y_std + y_mean
    err_before = np.sqrt(np.mean(va.Y**2, axis=0))
    err_after = np.sqrt(np.mean((va.Y - pred) ** 2, axis=0))
    stats = {
        "val_loss": best,
        "n_train": len(tr),
        "n_val": len(va),
        "rms_p_before_mm": float(err_before[2] * 1e3),
        "rms_p_after_mm": float(err_after[2] * 1e3),
    }
    return model, stats
