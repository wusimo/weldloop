"""Tests for the R4 task-planning layer.

Three things are worth asserting about putting a language model in a welding
cell, and none of them is "the model gave a good answer":

1. **Nothing it says reaches an actuator unchecked.**  Every setpoint goes
   through the clamp, and what was refused is recorded.
2. **It is not in the loop, structurally.**  The plan is read by a table
   lookup; the backend is called before the arc and never again.
3. **The layer is allowed to fail.**  No key, no network, a stale recording,
   a malformed answer — the cell falls back to rules and says so.

Plus the claim the demo is built on: what a plan buys on a stepped plate is
the *thickness note*, not the model's arithmetic — so the ablation that keeps
the plan and removes the note has to lose.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from weldloop.config import default_config
from weldloop.control.motion_layer import AdaptiveController
from weldloop.control.setpoints import Nominal
from weldloop.metrics import score
from weldloop.planning.job import (
    DEMO_JOB, FitupScan, demo_config, demo_seam, demo_seam_description,
)
from weldloop.planning.prompt import PLAN_SCHEMA, SYSTEM_PROMPT, request_digest
from weldloop.planning.schedule import PlanSchedule
from weldloop.planning.task_planner import SeamDescription, TaskPlanner
from weldloop.planning.vla import RecordedBackend, VLATaskPlanner
from weldloop.sim.runner import simulate
from weldloop.sim.seam import make_seam

SCAN = Path(__file__).resolve().parents[1] / "docs" / "vla_fitup_scan.png"


@pytest.fixture(scope="module")
def cfg():
    return demo_config()


@pytest.fixture(scope="module")
def seam(cfg):
    return demo_seam(cfg)


@pytest.fixture(scope="module")
def planned(cfg):
    """The recorded plan for the demo job, and its trace."""
    planner = VLATaskPlanner(cfg, backend=RecordedBackend())
    return planner.plan(DEMO_JOB, FitupScan(SCAN), demo_seam_description(cfg))


# --------------------------------------------------------------------------
class TestSeamThickness:
    """A stepped plate is a step, not a ramp."""

    def test_uniform_is_the_default(self):
        c = default_config()
        s = make_seam(c.seam, 0, thickness=c.joint.thickness)
        assert np.allclose(s.thickness, c.joint.thickness)

    def test_step_down_has_exactly_two_thicknesses(self, cfg, seam):
        assert sorted(np.unique(seam.thickness)) == pytest.approx(
            [cfg.seam.thickness_thin, cfg.joint.thickness]
        )

    def test_the_step_is_not_smoothed(self, cfg, seam):
        """Interpolating across the step would hand the cell a ramp the plate
        does not have, and the burn-through trip would be late everywhere."""
        s_step = cfg.seam.thickness_step_at * cfg.seam.length
        just_before = seam.thickness_at(s_step - 1.0e-3)
        just_after = seam.thickness_at(s_step + 1.0e-3)
        assert just_before == pytest.approx(cfg.joint.thickness)
        assert just_after == pytest.approx(cfg.seam.thickness_thin)

    def test_the_plant_sees_the_step(self, cfg, seam):
        from weldloop.control.baseline import BaselineController

        res = simulate(cfg, seed=0, controller=BaselineController(cfg), seam=seam)
        assert len(np.unique(res.table["truth_thickness"])) == 2


# --------------------------------------------------------------------------
class TestNothingReachesTheCellUnchecked:
    """The clamp is the door, and it keeps a receipt."""

    def test_absurd_current_is_clamped_and_reported(self, cfg):
        nom, reports = Nominal(
            I_set=900.0, v_travel=4.5e-3, weave_amp=0.0,
            p_target=4.0e-3, p_lo=3.0e-3, p_hi=5.0e-3, thickness=6.0e-3,
        ).clamped(cfg)
        assert nom.I_set == cfg.control.I_max_cmd
        assert [r.field for r in reports] == ["I_set"]
        assert reports[0].asked == 900.0

    def test_a_band_deeper_than_the_plate_is_refused(self, cfg):
        """The single most dangerous thing a planner can get wrong: carrying
        the thick section's band onto the thin section."""
        nom, reports = Nominal(
            I_set=200.0, v_travel=4.0e-3, weave_amp=1.0e-3,
            p_target=4.0e-3, p_lo=3.0e-3, p_hi=5.0e-3, thickness=4.0e-3,
        ).clamped(cfg)
        assert nom.p_hi <= cfg.control.bt_margin * 4.0e-3 + 1e-12
        assert nom.p_target <= nom.p_hi
        assert {r.field for r in reports} >= {"p_hi"}

    def test_implausible_thickness_is_clamped(self, cfg):
        nom, reports = Nominal(
            I_set=200.0, v_travel=4.0e-3, weave_amp=0.0,
            p_target=2.0e-3, p_lo=1.0e-3, p_hi=3.0e-3, thickness=0.2e-3,
        ).clamped(cfg)
        assert nom.thickness == Nominal.THICKNESS_MIN
        assert "thickness" in {r.field for r in reports}

    def test_a_clean_plan_is_clamped_nowhere(self, planned, cfg):
        _, trace = planned
        assert trace.n_clamped == 0, trace.clamp_lines


# --------------------------------------------------------------------------
class TestPlanSchedule:
    def test_lookup_covers_the_whole_seam(self, cfg):
        plan = TaskPlanner(cfg).plan(SeamDescription(length=0.2, thickness=6.0e-3))
        sch = PlanSchedule.from_plan(plan, cfg)
        for s in (-1.0, 0.0, 0.1, 0.2, 5.0):
            assert 0 <= sch.index_at(s) < len(sch.nominals)

    def test_boundaries_are_monotone(self, planned, cfg):
        plan, _ = planned
        sch = PlanSchedule.from_plan(plan, cfg)
        assert np.all(np.diff(sch.edges) > 0)

    def test_an_empty_plan_is_refused(self, cfg):
        plan = TaskPlanner(cfg).plan(SeamDescription(length=0.2, thickness=6.0e-3))
        with pytest.raises(ValueError):
            PlanSchedule.from_plan(replace(plan, segments=()), cfg)


# --------------------------------------------------------------------------
class TestTheRecording:
    def test_the_shipped_recording_matches_the_shipped_request(self, cfg, planned):
        _, trace = planned
        assert trace.fallback == "", trace.fallback
        assert trace.backend == "recorded"
        assert trace.digest == request_digest(
            DEMO_JOB.text, cfg, cfg.seam.length, SCAN.read_bytes()
        )

    def test_a_changed_request_refuses_to_replay(self, cfg):
        """A plan written for a different joint is worse than no plan."""
        other = replace(DEMO_JOB, text=DEMO_JOB.text + "\n附加: 改为 10 mm 板。")
        planner = VLATaskPlanner(cfg, backend=RecordedBackend())
        plan, trace = planner.plan(other, FitupScan(SCAN), demo_seam_description(cfg))
        assert trace.fallback.startswith("FileNotFoundError")
        assert trace.backend.endswith("->rules")
        assert len(plan.segments) >= 1  # the cell still got a plan

    def test_the_prompt_is_part_of_the_fingerprint(self, cfg):
        a = request_digest(DEMO_JOB.text, cfg, cfg.seam.length, b"x")
        b = request_digest(DEMO_JOB.text + " ", cfg, cfg.seam.length, b"x")
        assert a != b

    def test_the_schema_forbids_inventing_fields(self):
        assert PLAN_SCHEMA["additionalProperties"] is False
        assert PLAN_SCHEMA["properties"]["segments"]["items"][
            "additionalProperties"] is False
        assert "实时" in SYSTEM_PROMPT  # it is told, in the prompt, to stay offline


# --------------------------------------------------------------------------
class TestTheModelReadTheDocuments:
    """Assertions about the plan's *shape*, not about its taste in currents."""

    def test_it_cut_the_seam_more_than_once(self, planned):
        plan, _ = planned
        assert len(plan.segments) >= 3

    def test_segments_tile_the_seam(self, planned, cfg):
        plan, _ = planned
        assert plan.segments[0].s_start == pytest.approx(0.0)
        assert plan.segments[-1].s_end == pytest.approx(cfg.seam.length)
        for a, b in zip(plan.segments, plan.segments[1:]):
            assert a.s_end == pytest.approx(b.s_start)

    def test_the_thin_section_is_declared_thin(self, planned, cfg):
        """The thickness step is in the job card and the drawing only."""
        plan, _ = planned
        s_step = cfg.seam.thickness_step_at * cfg.seam.length
        after = [s for s in plan.segments if s.s_start >= s_step - 1e-9]
        assert after, "no segment starts at or after the thickness step"
        for seg in after:
            assert seg.thickness == pytest.approx(cfg.seam.thickness_thin)

    def test_a_segment_boundary_lands_on_the_step(self, planned, cfg):
        s_step = cfg.seam.thickness_step_at * cfg.seam.length
        plan, _ = planned
        edges = [s.s_start for s in plan.segments]
        assert min(abs(e - s_step) for e in edges) < 5.0e-3

    def test_the_band_follows_the_acceptance_rule(self, planned, cfg):
        """The rule is a fraction of thickness; copying the thick section's
        millimetres onto 4 mm plate is the failure this catches."""
        plan, _ = planned
        for seg in plan.segments:
            h = seg.thickness or cfg.joint.thickness
            assert seg.p_lo == pytest.approx(DEMO_JOB.p_frac_lo * h, rel=0.15)
            assert seg.p_hi <= DEMO_JOB.p_frac_hi * h * 1.05


# --------------------------------------------------------------------------
class TestTheLayerIsNotInTheLoop:
    """The architectural claim, as an assertion rather than a paragraph."""

    def test_the_backend_is_called_once_and_never_during_the_arc(self, cfg, seam):
        class CountingBackend:
            name, model = "counting", "test"

            def __init__(self, inner):
                self.inner, self.calls = inner, 0

            def complete(self, *args, **kwargs):
                self.calls += 1
                return self.inner.complete(*args, **kwargs)

        backend = CountingBackend(RecordedBackend())
        plan, _ = VLATaskPlanner(cfg, backend=backend).plan(
            DEMO_JOB, FitupScan(SCAN), demo_seam_description(cfg)
        )
        assert backend.calls == 1
        simulate(cfg, seed=0, controller=AdaptiveController(cfg, plan=plan), seam=seam)
        assert backend.calls == 1, "the planner was called while the arc was lit"

    def test_the_loop_decides_far_more_often_than_the_plan_says_anything(
        self, cfg, seam, planned
    ):
        plan, _ = planned
        ctl = AdaptiveController(cfg, plan=plan, name="vla+adapt")
        simulate(cfg, seed=0, controller=ctl, seam=seam)
        assert len(ctl.history) > 100 * len(plan.segments)

    def test_a_plan_is_read_by_table_lookup(self, planned, cfg):
        """No I/O, no model, no allocation worth mentioning: the real-time
        path is a searchsorted."""
        plan, _ = planned
        sch = PlanSchedule.from_plan(plan, cfg)
        before = sch.nominal_at(0.01)
        after = sch.nominal_at(cfg.seam.length - 0.01)
        assert before.thickness != after.thickness


# --------------------------------------------------------------------------
class TestTheThicknessNoteIsWhatBuysTheResult:
    """The ablation.  Same model, same segmentation, same setpoints; the only
    difference is whether one sentence of prose reached the safety monitor.

    If this ever fails in the direction of "the ablation is just as good", the
    demo's headline claim is wrong and the README has to change, not the test.
    """

    @pytest.fixture(scope="class")
    def arms(self, cfg, seam):
        planner = VLATaskPlanner(cfg, backend=RecordedBackend())
        plan, _ = planner.plan(DEMO_JOB, FitupScan(SCAN), demo_seam_description(cfg))
        blind = replace(plan, segments=tuple(
            replace(g, thickness=cfg.joint.thickness, p_target=cfg.control.p_target,
                    p_lo=cfg.control.p_lo, p_hi=cfg.control.p_hi)
            for g in plan.segments
        ))
        out = {}
        for name, pl in (("vla", plan), ("ablation", blind)):
            res = simulate(cfg, seed=0, seam=seam,
                           controller=AdaptiveController(cfg, plan=pl, name=name))
            out[name] = score(res.table, cfg, name)
        return out

    def test_reading_the_note_removes_most_of_the_burn_through(self, arms):
        assert arms["vla"].burn_through_length_mm < 0.2 * (
            arms["ablation"].burn_through_length_mm
        )

    def test_reading_the_note_keeps_penetration_in_a_band_that_moves(self, arms):
        assert arms["vla"].penetration_in_band_pct > (
            arms["ablation"].penetration_in_band_pct + 20.0
        )

    def test_and_it_is_not_paid_for_in_cycle_time(self, arms):
        """Being right about the plate is not the same as being slow."""
        assert arms["vla"].duration_s <= arms["ablation"].duration_s * 1.05
