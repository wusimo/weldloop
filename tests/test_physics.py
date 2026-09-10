"""Phase 1 tests: heat sources, arc characteristic, reduced-order melt pool.

These are *sanity* tests, not validation against experiment.  They assert the
qualitative behaviour a welding engineer would insist on (more power -> hotter,
faster travel -> less penetration, a wider gap -> burn-through) plus internal
consistency of the ODE with its own steady state.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from weldloop.config import TransferMode, default_config
from weldloop.physics import arc as arcmod
from weldloop.physics import heat_source as hs
from weldloop.physics.melt_pool import MeltPoolModel, PoolInputs, PoolState


@pytest.fixture(scope="module")
def cfg():
    return default_config()


@pytest.fixture(scope="module")
def model(cfg):
    return MeltPoolModel(cfg)


def _inputs(cfg, *, I=230.0, v=4.5e-3, gap=0.0, weave=0.0, v_wire=None):
    """Self-consistent inputs: the voltage follows the arc characteristic and
    the wire feed follows the burn-off law for the requested current."""
    V = arcmod.arc_voltage(I, cfg.arc.L_arc_ref, 15.0e-3, cfg.arc, cfg.consumable)
    if v_wire is None:
        v_wire = arcmod.melting_rate(I, 15.0e-3, cfg.arc)
    return PoolInputs(
        I=I,
        V=V,
        v_travel=v,
        v_wire=v_wire,
        gap=gap,
        thickness=cfg.joint.thickness,
        weave_amp=weave,
    )


# ==========================================================================
# heat sources
# ==========================================================================
class TestGoldak:
    def test_peak_is_at_the_arc_and_positive(self):
        g = hs.GoldakParams()
        q0 = float(hs.goldak_flux(0.0, 0.0, 0.0, 5000.0, g))
        assert q0 > 0.0
        for pt in [(g.a_f, 0.0, 0.0), (0.0, g.b, 0.0), (0.0, 0.0, g.c)]:
            assert float(hs.goldak_flux(*pt, 5000.0, g)) < q0

    def test_nothing_above_the_plate_surface(self):
        g = hs.GoldakParams()
        assert float(hs.goldak_flux(0.0, 0.0, -1.0e-3, 5000.0, g)) == 0.0

    def test_rear_lobe_is_longer_than_the_front_lobe(self):
        """The double ellipsoid exists precisely to be asymmetric."""
        g = hs.GoldakParams()
        d = 5.0e-3
        ahead = float(hs.goldak_flux(d, 0.0, 0.0, 5000.0, g))
        behind = float(hs.goldak_flux(-d, 0.0, 0.0, 5000.0, g))
        assert behind > ahead

    def test_integrates_to_the_total_power(self):
        """Normalisation check: the two half-ellipsoids must sum to Q.

        Each lobe is integrated on its own grid, because the front lobe is
        four times shorter than the rear one and a single grid fine enough for
        both is needlessly large.
        """
        g = hs.GoldakParams()
        Q = 5000.0
        y = np.linspace(-6.0 * g.b, 6.0 * g.b, 161)
        z = np.linspace(0.0, 6.0 * g.c, 161)
        total = 0.0
        for lo, hi in ((-6.0 * g.a_r, -1e-12), (1e-12, 6.0 * g.a_f)):
            xi = np.linspace(lo, hi, 161)
            X, Y, Z = np.meshgrid(xi, y, z, indexing="ij")
            q = hs.goldak_flux(X, Y, Z, Q, g)
            total += np.trapz(np.trapz(np.trapz(q, z, axis=2), y, axis=1), xi)
        assert total == pytest.approx(Q, rel=0.01)


class TestRosenthal:
    def test_trailing_centerline_matches_the_closed_form(self, cfg):
        d = 4.0e-3
        Q = 5000.0
        direct = float(hs.rosenthal_thick_plate(-d, 0.0, 0.0, Q, 5.0e-3, cfg.material))
        closed = float(hs.rosenthal_trailing_centerline(d, Q, cfg.material))
        assert direct == pytest.approx(closed, rel=1e-9)

    def test_peak_temperature_grows_with_power(self, cfg):
        y = 3.0e-3
        temps = [
            hs._max_T_at(y, 0.0, Q, 5.0e-3, cfg.material) for Q in (3000.0, 5000.0, 7000.0)
        ]
        assert temps[0] < temps[1] < temps[2]

    def test_peak_temperature_falls_with_travel_speed(self, cfg):
        y = 3.0e-3
        temps = [
            hs._max_T_at(y, 0.0, 5000.0, v, cfg.material)
            for v in (3.0e-3, 6.0e-3, 12.0e-3)
        ]
        assert temps[0] > temps[1] > temps[2]

    def test_fusion_zone_grows_with_power_and_shrinks_with_speed(self, cfg):
        m = cfg.material
        w_lo = hs.melt_isotherm_halfwidth(4000.0, 5.0e-3, m)
        w_hi = hs.melt_isotherm_halfwidth(6000.0, 5.0e-3, m)
        w_fast = hs.melt_isotherm_halfwidth(4000.0, 10.0e-3, m)
        assert w_hi > w_lo > w_fast > 0.0
        d_lo = hs.melt_isotherm_depth(4000.0, 5.0e-3, m)
        d_hi = hs.melt_isotherm_depth(6000.0, 5.0e-3, m)
        assert d_hi > d_lo > 0.0

    def test_analytic_fusion_area_is_the_heat_input_relation(self, cfg):
        a = hs.fusion_area_from_heat_input(26.5, 230.0, 4.5e-3, 0.8, 0.5, cfg.material)
        b = hs.fusion_area_from_heat_input(26.5, 230.0, 9.0e-3, 0.8, 0.5, cfg.material)
        assert a == pytest.approx(2.0 * b, rel=1e-9)
        assert 10.0e-6 < a < 200.0e-6  # a plausible bead, tens of mm^2


# ==========================================================================
# arc
# ==========================================================================
class TestArc:
    def test_arc_characteristic_round_trip(self, cfg):
        V = arcmod.arc_voltage(230.0, 5.0e-3, 15.0e-3, cfg.arc, cfg.consumable)
        L = arcmod.arc_length_from_voltage(V, 230.0, 15.0e-3, cfg.arc, cfg.consumable)
        assert L == pytest.approx(5.0e-3, rel=1e-9)

    def test_voltage_rises_with_arc_length_and_with_stickout(self, cfg):
        base = arcmod.arc_voltage(230.0, 5.0e-3, 15.0e-3, cfg.arc, cfg.consumable)
        assert arcmod.arc_voltage(230.0, 7.0e-3, 15.0e-3, cfg.arc, cfg.consumable) > base
        assert arcmod.arc_voltage(230.0, 5.0e-3, 20.0e-3, cfg.arc, cfg.consumable) > base

    def test_unknown_tip_wear_biases_the_arc_length_estimate(self, cfg):
        """A 10 % resistivity drift must look like an arc-length change."""
        V_true = arcmod.arc_voltage(
            230.0, 5.0e-3, 15.0e-3, cfg.arc, cfg.consumable, rho_scale=1.10
        )
        L_est = arcmod.arc_length_from_voltage(
            V_true, 230.0, 15.0e-3, cfg.arc, cfg.consumable
        )
        assert L_est > 5.0e-3
        assert abs(L_est - 5.0e-3) < 1.0e-3  # a bias, not a catastrophe

    def test_burn_off_law_round_trip(self, cfg):
        for I in (120.0, 230.0, 330.0):
            v = arcmod.melting_rate(I, 16.0e-3, cfg.arc)
            assert arcmod.current_for_melting_rate(v, 16.0e-3, cfg.arc) == pytest.approx(
                I, rel=1e-9
            )

    def test_longer_stickout_melts_faster_at_the_same_current(self, cfg):
        assert arcmod.melting_rate(230.0, 20.0e-3, cfg.arc) > arcmod.melting_rate(
            230.0, 12.0e-3, cfg.arc
        )

    def test_transfer_mode_thresholds(self, cfg):
        assert arcmod.classify_transfer_mode(120.0, cfg.arc) is TransferMode.SHORT_CIRCUIT
        assert arcmod.classify_transfer_mode(220.0, cfg.arc) is TransferMode.GLOBULAR
        assert arcmod.classify_transfer_mode(300.0, cfg.arc) is TransferMode.SPRAY

    def test_oscillation_frequency_falls_with_pool_width(self, cfg):
        f_small, _ = arcmod.pool_oscillation(8.0e-3, 4.0e-3, 0.2, cfg.arc, cfg.material)
        f_big, _ = arcmod.pool_oscillation(14.0e-3, 4.0e-3, 0.2, cfg.arc, cfg.material)
        assert f_small > f_big > 0.0

    def test_oscillation_amplitude_grows_with_depth_and_superheat(self, cfg):
        _, a_shallow = arcmod.pool_oscillation(11.0e-3, 3.0e-3, 0.2, cfg.arc, cfg.material)
        _, a_deep = arcmod.pool_oscillation(11.0e-3, 5.5e-3, 0.2, cfg.arc, cfg.material)
        _, a_hot = arcmod.pool_oscillation(11.0e-3, 3.0e-3, 0.6, cfg.arc, cfg.material)
        assert a_deep > a_shallow
        assert a_hot > a_shallow

    def test_pool_oscillation_is_in_a_measurable_band(self, cfg):
        """The whole sensing thesis needs f_osc inside the 5 kHz Nyquist band."""
        f, a = arcmod.pool_oscillation(11.0e-3, 4.5e-3, 0.25, cfg.arc, cfg.material)
        assert 20.0 < f < 0.5 * cfg.sensors.f_power
        assert cfg.arc.E_a * a > cfg.sensors.noise_V  # ripple beats per-sample noise

    def test_short_circuit_rate_rises_with_ripple_and_falls_with_arc_length(self, cfg):
        base = arcmod.short_circuit_rate(230.0, 5.0e-3, 0.25e-3, cfg.arc)
        assert arcmod.short_circuit_rate(230.0, 5.0e-3, 0.40e-3, cfg.arc) > base
        assert arcmod.short_circuit_rate(230.0, 8.0e-3, 0.25e-3, cfg.arc) < base

    def test_short_circuit_rate_peaks_in_short_arc_transfer(self, cfg):
        peak = arcmod.short_circuit_rate(cfg.arc.I_sc_peak, 5.0e-3, 0.3e-3, cfg.arc)
        assert peak > arcmod.short_circuit_rate(320.0, 5.0e-3, 0.3e-3, cfg.arc)
        assert peak > arcmod.short_circuit_rate(40.0, 5.0e-3, 0.3e-3, cfg.arc)

    def test_short_circuit_process_is_deterministic_and_rate_ordered(self, cfg):
        def duty(rate: float, seed: int) -> float:
            proc = arcmod.ShortCircuitProcess(cfg.arc, np.random.default_rng(seed))
            n = 200_000
            hits = sum(proc.step(2.0e-4, rate) for _ in range(n))
            return hits / n

        assert duty(30.0, 7) == duty(30.0, 7)
        assert duty(60.0, 7) > duty(15.0, 7)


# ==========================================================================
# melt pool
# ==========================================================================
class TestMeltPoolSteadyState:
    def test_nominal_operating_point_is_a_plausible_bead(self, cfg, model):
        s = model.steady_state(_inputs(cfg))
        assert cfg.material.T_m < s.T_pool < 2600.0
        assert 6.0e-3 < s.w < 20.0e-3
        assert 2.0e-3 < s.p < cfg.joint.thickness
        assert not model.defects(s, _inputs(cfg)).burn_through

    def test_penetration_grows_with_current(self, cfg, model):
        ps = [model.steady_state(_inputs(cfg, I=I)).p for I in (180.0, 230.0, 280.0)]
        assert ps[0] < ps[1] < ps[2]

    def test_penetration_falls_with_travel_speed(self, cfg, model):
        ps = [
            model.steady_state(_inputs(cfg, v=v)).p for v in (3.0e-3, 4.5e-3, 9.0e-3)
        ]
        assert ps[0] > ps[1] > ps[2]

    def test_pool_width_falls_with_travel_speed(self, cfg, model):
        ws = [
            model.steady_state(_inputs(cfg, v=v)).w for v in (3.0e-3, 4.5e-3, 9.0e-3)
        ]
        assert ws[0] > ws[1] > ws[2]

    def test_a_root_gap_deepens_and_narrows_the_pool(self, cfg, model):
        """The central disturbance: a gap removes the heat sink and lets the
        arc root descend, so the same parameters dig deeper."""
        states = [model.steady_state(_inputs(cfg, gap=g)) for g in (0.0, 2.0e-3, 4.0e-3)]
        assert states[0].p < states[1].p < states[2].p
        assert states[0].w > states[1].w > states[2].w
        assert states[0].T_pool < states[1].T_pool < states[2].T_pool

    def test_weaving_widens_and_shallows_the_bead(self, cfg, model):
        flat = model.steady_state(_inputs(cfg, gap=4.0e-3, weave=0.0))
        woven = model.steady_state(_inputs(cfg, gap=4.0e-3, weave=2.0e-3))
        assert woven.p < flat.p
        assert woven.w > flat.w

    def test_gap_fill_falls_as_the_gap_opens(self, cfg, model):
        fs = [model.steady_state(_inputs(cfg, gap=g)).f for g in (0.0, 2.0e-3, 4.0e-3)]
        assert fs[0] > fs[1] > fs[2]

    def test_more_wire_fills_more(self, cfg, model):
        slow = model.steady_state(_inputs(cfg, gap=4.0e-3, v_wire=5.0 / 60.0))
        fast = model.steady_state(_inputs(cfg, gap=4.0e-3, v_wire=9.0 / 60.0))
        assert fast.f > slow.f

    def test_equilibrium_reproduces_its_own_fusion_area_relation(self, cfg, model):
        """At equilibrium A must equal q_fus / (rho * h_m * v) — the textbook
        melting-efficiency relation, with the *realised* efficiency."""
        u = _inputs(cfg)
        s = model.steady_state(u)
        d = model.derived(s, u)
        A_direct = math.pi / 4.0 * s.w * s.p
        A_relation = d.q_fus / (cfg.material.rho * cfg.material.h_m * u.v_travel)
        assert A_direct == pytest.approx(A_relation, rel=2e-3)
        eta_eff = model.effective_melting_efficiency(s, u)
        assert 0.0 < eta_eff < cfg.pool.eta_melt  # losses are paid before melting

    def test_effective_melting_efficiency_is_in_a_credible_range(self, cfg, model):
        u = _inputs(cfg)
        s = model.steady_state(u)
        assert 0.25 < model.effective_melting_efficiency(s, u) < 0.70


class TestMeltPoolDefects:
    def test_wide_gap_burns_through_at_baseline_parameters(self, cfg, model):
        u = _inputs(cfg, gap=4.0e-3)
        s = model.steady_state(u)
        flags = model.defects(s, u)
        assert flags.burn_through
        assert flags.burn_through_bridging  # drops out before melting the full plate

    def test_burn_through_when_penetration_exceeds_thickness(self, cfg, model):
        u = _inputs(cfg)
        deep = PoolState(T_pool=2200.0, w=12.0e-3, p=cfg.joint.thickness * 1.01, f=1.0)
        assert model.defects(deep, u).burn_through_thickness

    def test_nominal_zero_gap_is_defect_free(self, cfg, model):
        u = _inputs(cfg)
        flags = model.defects(model.steady_state(u), u)
        assert not flags.burn_through
        assert not flags.lack_of_fusion

    def test_cold_process_lacks_fusion(self, cfg, model):
        u = _inputs(cfg, I=130.0)
        flags = model.defects(model.steady_state(u), u)
        assert flags.lack_of_fusion and flags.lof_cold

    def test_fast_travel_over_a_wide_gap_underfills(self, cfg, model):
        u = _inputs(cfg, v=12.0e-3, gap=4.0e-3)
        flags = model.defects(model.steady_state(u), u)
        assert flags.lack_of_fusion
        assert flags.lof_underfill or flags.lof_sidewall

    def test_burn_through_risk_is_monotone_in_gap(self, cfg, model):
        margins = []
        for g in (0.0, 1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3):
            u = _inputs(cfg, gap=g)
            fl = model.defects(model.steady_state(u), u)
            margins.append(fl.w_root - fl.w_crit)
        assert all(b > a for a, b in zip(margins, margins[1:]))
        assert margins[0] < 0.0 < margins[-1]


class TestMeltPoolDynamics:
    def test_step_is_deterministic(self, cfg, model):
        u = _inputs(cfg)
        s0 = model.initial_state()
        a = model.step(s0, u, cfg.sim.dt)[0].to_array()
        b = model.step(s0, u, cfg.sim.dt)[0].to_array()
        assert np.array_equal(a, b)

    def test_cold_start_converges_to_the_steady_state(self, cfg, model):
        u = _inputs(cfg)
        target = model.steady_state(u)
        s = model.initial_state()
        for _ in range(int(6.0 / cfg.sim.dt)):
            s, _ = model.step(s, u, cfg.sim.dt)
        assert s.p == pytest.approx(target.p, rel=0.02)
        assert s.w == pytest.approx(target.w, rel=0.02)
        assert s.T_pool == pytest.approx(target.T_pool, rel=0.01)

    def test_states_stay_finite_and_clamped_under_abuse(self, cfg, model):
        """Zero current, absurd wire feed, huge gap: still no NaN, still bounded."""
        rng = np.random.default_rng(0)
        s = model.initial_state()
        for _ in range(20_000):
            u = PoolInputs(
                I=float(rng.uniform(0.0, 400.0)),
                V=float(rng.uniform(0.0, 45.0)),
                v_travel=float(rng.uniform(1.0e-3, 20.0e-3)),
                v_wire=float(rng.uniform(0.0, 0.30)),
                gap=float(rng.uniform(0.0, 8.0e-3)),
                thickness=cfg.joint.thickness,
                weave_amp=float(rng.uniform(0.0, 4.0e-3)),
            )
            s, _ = model.step(s, u, cfg.sim.dt)
            assert np.all(np.isfinite(s.to_array()))
        assert cfg.material.T_m <= s.T_pool <= 4000.0
        assert 0.0 < s.w <= 0.10 and 0.0 < s.p <= 0.10
        assert 0.0 <= s.f <= cfg.pool.f_max

    def test_turning_the_arc_off_collapses_the_pool(self, cfg, model):
        u = PoolInputs(
            I=0.0, V=0.0, v_travel=4.5e-3, v_wire=0.0,
            gap=0.0, thickness=cfg.joint.thickness,
        )
        s = model.steady_state(_inputs(cfg))
        for _ in range(int(3.0 / cfg.sim.dt)):
            s, _ = model.step(s, u, cfg.sim.dt)
        assert s.p < 1.0e-3
        assert s.T_pool == pytest.approx(cfg.material.T_m)

    def test_penetration_responds_within_the_control_bandwidth(self, cfg, model):
        """A step in gap must move penetration on a timescale the 20 ms motion
        layer can act on — otherwise adaptive control is pointless."""
        u0 = _inputs(cfg, gap=0.0)
        u1 = _inputs(cfg, gap=4.0e-3)
        s = model.steady_state(u0)
        p0, p_inf = s.p, model.steady_state(u1).p
        t = 0.0
        while s.p < p0 + 0.63 * (p_inf - p0) and t < 5.0:
            s, _ = model.step(s, u1, cfg.sim.dt)
            t += cfg.sim.dt
        assert 0.02 < t < 2.0

    def test_array_wrapper_matches_the_dataclass_path(self, cfg, model):
        u = _inputs(cfg)
        s = model.steady_state(u)
        direct = model.step(s, u, cfg.sim.dt)[0].to_array()
        via_array = model.step_array(s.to_array(), u.to_array(), cfg.sim.dt)
        assert np.allclose(direct, via_array, rtol=0, atol=0)
