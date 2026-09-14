"""R4: a vision-language model doing the task-planning layer, and only that.

This is the VLA-shaped part of the demo, put where the roadmap says it belongs
and nowhere else.  The model is handed the two documents a welding shop
actually has — a job card in prose and a fit-up inspection *image* — and asked
for one thing: cut the seam into segments and give each one a starting point
and an acceptance band.  It is given seconds.  It never sees the arc.

Why this and not torch control:

* **Latency.** The call takes seconds.  The motion layer decides every 20 ms
  and burn-through develops in under a second.  ``PlannerTrace.latency_s``
  measures it against ``control.dt`` on every run, so the ratio in the README
  is not an argument, it is a logged number.
* **Visibility.** Its natural real-time input is a camera, and the RGB arm in
  this repo measures what the arc does to one.
* **Accountability.** Every number the model returns goes through
  :meth:`Nominal.clamped` before the arc is lit, and what it asked for that it
  did not get is recorded.  A plan is a table; a table can be reviewed,
  diffed, signed and archived.  A policy that emits torch commands at 50 Hz
  cannot be any of those things with today's tools.

Two backends, because the repo has to run with no network and no key:

``ClaudeBackend``
    The real call: ``claude-opus-5``, the scan image as a base64 block, the
    plan schema as a structured-output format.
``RecordedBackend``
    Replays a response saved under ``data/vla/``, keyed by a digest of the
    exact request.  Change the prompt, the schema, the job card, the machine
    limits or the image and the digest changes, so a stale recording refuses
    to load rather than quietly answering a question nobody asked.

TODO(real-hw): a third backend belongs here eventually — the customer's own
model behind their own endpoint.  The interface is one method.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from weldloop.config import WeldConfig, default_config
from weldloop.planning.job import FitupScan, JobCard
from weldloop.planning.prompt import (
    PLAN_SCHEMA,
    SYSTEM_PROMPT,
    build_user_text,
    request_digest,
)
from weldloop.planning.schedule import PlanSchedule
from weldloop.planning.task_planner import Segment, SeamDescription, TaskPlanner, WeldPlan
from weldloop.physics.arc import melting_rate

__all__ = [
    "PlannerTrace", "PlannerBackend", "ClaudeBackend", "RecordedBackend",
    "VLATaskPlanner", "RECORDING_DIR", "MODEL",
]

#: The model this layer is written against.
MODEL = "claude-opus-5"

RECORDING_DIR = Path(__file__).resolve().parents[2] / "data" / "vla"


# --------------------------------------------------------------------------
# what a planning call produced, including everything that went wrong
# --------------------------------------------------------------------------
@dataclass(slots=True)
class PlannerTrace:
    """The audit record of one planning call."""

    backend: str
    model: str
    digest: str
    latency_s: float = 0.0
    n_segments: int = 0
    n_clamped: int = 0
    clamp_lines: tuple[str, ...] = ()
    reading: str = ""
    risks: tuple[str, ...] = ()
    fallback: str = ""          # non-empty: the rule planner produced this
    raw: dict = field(default_factory=dict)

    def cycles_missed(self, cfg: WeldConfig) -> float:
        """How many motion-layer decisions go by while this call is in flight.

        The number this demo exists to make concrete.
        """
        return self.latency_s / cfg.control.dt

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model,
            "digest": self.digest,
            "latency_s": round(self.latency_s, 3),
            "n_segments": self.n_segments,
            "n_clamped": self.n_clamped,
            "clamp_lines": list(self.clamp_lines),
            "reading": self.reading,
            "risks": list(self.risks),
            "fallback": self.fallback,
        }


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
class PlannerBackend(Protocol):
    """One method: documents in, plan dict out."""

    name: str
    model: str

    def complete(self, system: str, user_text: str, image: bytes | None,
                 schema: dict) -> dict: ...


class ClaudeBackend:
    """The real call.  Needs ``anthropic`` installed and a credential.

    Nothing in the repository's test suite touches this: tests must run with
    no network, and a test whose result depends on a model call is not a test
    of the welding code.
    """

    name = "claude"

    def __init__(self, model: str = MODEL, *, max_tokens: int = 16000) -> None:
        import anthropic  # noqa: F401  (import here so the dep stays optional)

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user_text: str, image: bytes | None,
                 schema: dict) -> dict:
        content: list[dict] = []
        if image is not None:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(image).decode("utf-8"),
                    },
                }
            )
        content.append({"type": "text", "text": user_text})

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


class RecordedBackend:
    """Replays a saved response for this exact request.

    The recording carries the digest of the request it answered.  If the
    request has changed, this raises instead of replaying: a plan that was
    written for a different joint is worse than no plan.
    """

    name = "recorded"

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = Path(directory) if directory is not None else RECORDING_DIR
        self.model = MODEL
        self.source: Path | None = None
        self.recorded_at = ""

    def path_for(self, digest: str) -> Path:
        return self.dir / f"plan_{digest}.json"

    def complete(self, system: str, user_text: str, image: bytes | None,
                 schema: dict) -> dict:
        digest = _digest_of(system, user_text, image, schema)
        path = self.path_for(digest)
        if not path.is_file():
            raise FileNotFoundError(
                f"no recorded plan for request {digest}. The job card, the "
                f"prompt, the schema, the machine limits or the scan image "
                f"changed since the recording was made; re-record with "
                f"scripts/vla_plan.py --live, or fall back to the rule planner."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.source = path
        self.model = payload.get("model", MODEL)
        self.recorded_at = payload.get("recorded_at", "")
        return payload["plan"]


def _digest_of(system: str, user_text: str, image: bytes | None, schema: dict) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(system.encode())
    h.update(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode())
    h.update(user_text.encode())
    if image is not None:
        h.update(hashlib.sha256(image).digest())
    return h.hexdigest()[:16]


def resolve_backend(prefer_live: bool = False) -> PlannerBackend:
    """Pick a backend.

    Live only when explicitly asked for *and* possible.  The default is the
    recording, so ``pytest -q`` and a fresh clone behave the same on a machine
    with no key and no network.
    """
    if prefer_live:
        try:
            return ClaudeBackend()
        except Exception as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                f"live planning backend unavailable ({exc}). Install "
                f"`anthropic` and set ANTHROPIC_API_KEY, or drop --live."
            ) from exc
    return RecordedBackend()


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------
class VLATaskPlanner:
    """Task planning by a vision-language model, with a rule-based floor.

    The fallback is not politeness.  This layer is allowed to fail — no key,
    no network, a malformed answer, a recording that no longer matches — and
    a welding cell that stops because a model was unreachable is not a product.
    On any failure it returns the rule planner's plan and says so in the trace.
    """

    def __init__(self, cfg: WeldConfig | None = None,
                 backend: PlannerBackend | None = None) -> None:
        self.cfg = cfg or default_config()
        self.backend = backend if backend is not None else resolve_backend()
        self.rules = TaskPlanner(self.cfg)

    def plan(
        self,
        job: JobCard,
        scan: FitupScan,
        seam_desc: SeamDescription,
        *,
        measured=None,
    ) -> tuple[WeldPlan, PlannerTrace]:
        """Produce a plan and the audit record of how it was produced."""
        cfg = self.cfg
        image = scan.path.read_bytes() if scan.exists() else None
        user_text = build_user_text(job.text, cfg, seam_desc.length)
        digest = request_digest(job.text, cfg, seam_desc.length, image)
        trace = PlannerTrace(
            backend=getattr(self.backend, "name", "?"),
            model=getattr(self.backend, "model", "?"),
            digest=digest,
        )

        t0 = time.perf_counter()
        try:
            payload = self.backend.complete(SYSTEM_PROMPT, user_text, image, PLAN_SCHEMA)
            trace.latency_s = time.perf_counter() - t0
            plan = self._to_plan(payload, seam_desc)
            trace.reading = str(payload.get("reading", ""))
            trace.risks = tuple(str(r) for r in payload.get("risks", ()))
            trace.raw = payload
        except Exception as exc:
            trace.latency_s = time.perf_counter() - t0
            trace.fallback = f"{type(exc).__name__}: {exc}"
            plan = self.rules.plan(seam_desc, measured=measured)
            trace.backend = f"{trace.backend}->rules"

        schedule = PlanSchedule.from_plan(plan, cfg)
        trace.n_segments = len(plan.segments)
        trace.n_clamped = schedule.n_clamped
        trace.clamp_lines = tuple(schedule.clamp_lines())
        return plan, trace

    # -- payload -> WeldPlan ---------------------------------------------
    def _to_plan(self, payload: dict, seam_desc: SeamDescription) -> WeldPlan:
        """Convert the model's answer into a plan, in SI, sorted and gap-free.

        Deliberately strict.  Anything missing, non-numeric or non-monotone is
        an exception, which sends the caller to the rule planner — a plan the
        cell only half understands is the worst of the three outcomes.
        """
        cfg = self.cfg
        raw = payload["segments"]
        if not raw:
            raise ValueError("plan contains no segments")

        items = sorted(raw, key=lambda d: float(d["s_start_mm"]))
        stickout = cfg.robot.ctwd_nom - cfg.arc.L_arc_ref
        segments: list[Segment] = []
        cursor = 0.0
        for i, d in enumerate(items):
            s0 = cursor if i == 0 else max(float(d["s_start_mm"]) * 1e-3, cursor)
            s1 = float(d["s_end_mm"]) * 1e-3
            if i == len(items) - 1:
                s1 = seam_desc.length
            if s1 <= s0:
                raise ValueError(f"segment {i} is empty or out of order: {s0} -> {s1}")
            I = float(d["I_set_A"])
            segments.append(
                Segment(
                    s_start=s0,
                    s_end=min(s1, seam_desc.length),
                    gap_class=str(d["gap_class"]),
                    I_set=I,
                    v_travel=float(d["v_travel_mm_s"]) * 1e-3,
                    v_wire=melting_rate(I, stickout, cfg.arc),
                    weave_amp=float(d["weave_amp_mm"]) * 1e-3,
                    rationale=str(d["rationale"]),
                    thickness=float(d["thickness_mm"]) * 1e-3,
                    p_target=float(d["p_target_mm"]) * 1e-3,
                    p_lo=float(d["p_lo_mm"]) * 1e-3,
                    p_hi=float(d["p_hi_mm"]) * 1e-3,
                )
            )
            cursor = segments[-1].s_end

        cycle = sum((s.s_end - s.s_start) / s.v_travel for s in segments)
        return WeldPlan(
            segments=tuple(segments),
            p_target=cfg.control.p_target,
            p_lo=cfg.control.p_lo,
            p_hi=cfg.control.p_hi,
            estimated_cycle_time=cycle,
            assumptions=tuple(str(a) for a in payload.get("assumptions", ())),
        )


def save_recording(payload: dict, digest: str, model: str,
                   directory: Path | None = None, note: str = "") -> Path:
    """Write a response to ``data/vla/`` so the demo can be replayed offline."""
    import datetime as _dt

    directory = Path(directory) if directory is not None else RECORDING_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"plan_{digest}.json"
    path.write_text(
        json.dumps(
            {
                "digest": digest,
                "model": model,
                "recorded_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "note": note,
                "plan": payload,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
