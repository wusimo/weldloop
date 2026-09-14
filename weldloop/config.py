"""Configuration objects for the weldloop demo.

All quantities are SI unless the field name says otherwise
(``_mm``, ``_mm_s``, ``_m_min`` suffixes are convenience inputs that are
converted to SI by validators).

IMPORTANT — parameter provenance
--------------------------------
Every numeric default below is either

* a **textbook physical property of mild steel** quoted to two significant
  figures (density, specific heat, conductivity, melting point, latent heat),
  or
* a **lumped calibration constant** of the reduced-order model that has *no*
  independent physical meaning and was chosen so that the nominal operating
  point of this demo lands in a plausible bead geometry.

None of these numbers is claimed to come from a specific publication.  They
are all configurable and are expected to be re-identified from the real
phase-1 capture data.  See ``README.md`` section "走向真实硬件".
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

# --------------------------------------------------------------------------
# physical constants
# --------------------------------------------------------------------------
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant [W m^-2 K^-4]
G_GRAV = 9.80665  # standard gravity [m s^-2]


class TransferMode(str, Enum):
    """GMAW metal-transfer modes, ordered by increasing current."""

    SHORT_CIRCUIT = "short_circuit"
    GLOBULAR = "globular"
    SPRAY = "spray"


# --------------------------------------------------------------------------
# material / consumable
# --------------------------------------------------------------------------
class MaterialConfig(BaseModel):
    """Base-metal thermophysical properties.

    Defaults are typical **mild steel (Q235 / S235 / A36 class)** values,
    quoted at an effective high temperature.  CONFIGURABLE — re-identify for
    the customer's actual material.
    """

    name: str = "mild_steel"
    rho: float = Field(7800.0, description="density [kg/m^3]")
    c_p: float = Field(700.0, description="effective specific heat [J/(kg K)]")
    k: float = Field(30.0, description="effective thermal conductivity [W/(m K)]")
    T_m: float = Field(1800.0, description="melting temperature [K]")
    T_0: float = Field(300.0, description="ambient / preheat temperature [K]")
    L_f: float = Field(2.7e5, description="latent heat of fusion [J/kg]")
    emissivity: float = Field(0.6, description="pool surface emissivity [-]")
    gamma: float = Field(1.2, description="surface tension of the melt [N/m]")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m^2/s]."""
        return self.k / (self.rho * self.c_p)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def h_m(self) -> float:
        """Enthalpy required to melt unit mass from ``T_0`` [J/kg]."""
        return self.c_p * (self.T_m - self.T_0) + self.L_f

    @computed_field  # type: ignore[prop-decorator]
    @property
    def capillary_length(self) -> float:
        """Capillary length sqrt(gamma / (rho g)) [m] — surface-tension bridging scale."""
        return (self.gamma / (self.rho * G_GRAV)) ** 0.5


class ConsumableConfig(BaseModel):
    """Filler wire."""

    d_wire: float = Field(1.2e-3, description="wire diameter [m]")
    eta_dep: float = Field(0.95, description="deposition efficiency (1 - spatter loss) [-]")
    rho_e: float = Field(
        1.0e-6, description="effective electrical resistivity of hot wire [ohm m]"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def area(self) -> float:
        """Wire cross-sectional area [m^2]."""
        import math

        return math.pi * self.d_wire**2 / 4.0


# --------------------------------------------------------------------------
# joint
# --------------------------------------------------------------------------
class JointConfig(BaseModel):
    """Square-butt joint geometry.

    The demo simulates a **single-pass square-butt weld on medium-thick
    plate**.  Multi-pass thick-plate welding is the same control problem
    repeated per pass; nothing in the controller assumes a single pass.
    """

    thickness: float = Field(6.0e-3, description="plate thickness h [m]")
    A_reinf: float = Field(
        8.0e-6, description="target reinforcement (crown) cross-section [m^2]"
    )
    sidewall_margin: float = Field(
        1.5e-3,
        description="required pool half-overlap onto each sidewall beyond the gap [m]",
    )


# --------------------------------------------------------------------------
# reduced-order melt-pool model
# --------------------------------------------------------------------------
class PoolModelConfig(BaseModel):
    """Lumped coefficients of the reduced-order melt-pool ODE.

    LUMPED CALIBRATION CONSTANTS — not physical properties.  See
    ``physics/melt_pool.py`` for the equations each one appears in.
    """

    eta_arc: float = Field(0.80, description="arc efficiency, power into workpiece [-]")
    eta_melt: float = Field(
        0.50, description="melting efficiency: net power fraction that creates melt [-]"
    )
    C_cond: float = Field(1.70, description="conduction shape factor [-]")
    h_conv: float = Field(50.0, description="convective film coefficient [W/(m^2 K)]")
    kappa_L: float = Field(2.0, description="pool length / pool width [-]")

    # gap coupling ---------------------------------------------------------
    k_gap_cond: float = Field(
        0.90,
        description="how strongly a root gap removes the conduction heat sink [-]",
    )
    k_gap_ar: float = Field(
        2.20,
        description="root-gap digging strength: AR is divided by (1 + k*min(g/w,1)) [-]",
    )

    k_weave: float = Field(
        0.80,
        description="weaving widens and shallows the bead: AR *= (1 + k*2*a_weave/w) [-]",
    )

    # aspect-ratio law -----------------------------------------------------
    AR_0: float = Field(3.50, description="nominal bead aspect ratio width/depth [-]")
    I_ref: float = Field(220.0, description="reference current for the AR law [A]")
    v_ref: float = Field(5.0e-3, description="reference travel speed for the AR law [m/s]")
    n_I: float = Field(0.60, description="AR exponent on (I_ref/I) [-]")
    n_v: float = Field(0.20, description="AR exponent on (v_ref/v) [-]")
    AR_min: float = Field(1.20, description="aspect-ratio clip, low [-]")
    AR_max: float = Field(8.00, description="aspect-ratio clip, high [-]")
    ar_mode_gain: dict[TransferMode, float] = Field(
        default_factory=lambda: {
            TransferMode.SHORT_CIRCUIT: 1.25,  # wider, shallower
            TransferMode.GLOBULAR: 1.00,
            TransferMode.SPRAY: 0.80,  # narrower, deeper (finger penetration)
        },
        description="aspect-ratio multiplier per transfer mode [-]",
    )

    # time constants -------------------------------------------------------
    tau_A_max: float = Field(
        0.30,
        description="cap on the fusion-area relaxation time [s] (see melt_pool docstring)",
    )
    tau_A_min: float = Field(20.0e-3, description="floor on the fusion-area relaxation time [s]")
    tau_T_max: float = Field(
        0.25, description="cap on the pool-temperature relaxation time [s]"
    )
    tau_ar: float = Field(40.0e-3, description="aspect-ratio relaxation time [s]")
    tau_f: float = Field(80.0e-3, description="gap-fill relaxation time [s]")

    # numerical floors -----------------------------------------------------
    w_floor: float = Field(0.5e-3, description="numerical floor on pool width [m]")
    p_floor: float = Field(0.1e-3, description="numerical floor on penetration [m]")
    f_max: float = Field(3.0, description="clip on gap-fill ratio (>1 = over-fill) [-]")


class DefectConfig(BaseModel):
    """Thresholds for the burn-through / lack-of-fusion detectors."""

    beta_bt: float = Field(
        1.00, description="burn-through when p >= beta_bt * plate thickness [-]"
    )
    c_st: float = Field(
        2.05,
        description="capillary bridging coefficient: w_crit = c_st * capillary_length [-]",
    )
    k_fill_support: float = Field(
        0.30,
        description="deposited filler raises the bridging limit: w_crit *= (1 + k*min(f,1)) [-]",
    )
    min_hole_length: float = Field(
        0.20e-3,
        description=(
            "shortest run of the burn-through condition that counts as a hole [m]; "
            "shorter runs are marginal excursions of the bridging criterion"
        ),
    )
    p_min: float = Field(2.5e-3, description="minimum acceptable penetration [m]")
    f_min: float = Field(0.85, description="minimum acceptable gap-fill ratio [-]")


# --------------------------------------------------------------------------
# arc / power source
# --------------------------------------------------------------------------
class ArcConfig(BaseModel):
    """Static arc characteristic and metal-transfer thresholds.

    V = V_0 + E_a * L_arc + R_stickout(I) * I
    """

    V_0: float = Field(15.5, description="sum of anode + cathode falls [V]")
    E_a: float = Field(1600.0, description="arc column voltage gradient [V/m]")
    stickout: float = Field(15.0e-3, description="contact-tip-to-work minus arc length [m]")

    I_globular: float = Field(180.0, description="short-circuit -> globular threshold [A]")
    I_spray: float = Field(260.0, description="globular -> spray transition current [A]")

    # pool-surface oscillation --------------------------------------------
    # This is the mechanism that makes the pool observable from the power
    # source.  With the wire feed fixed, mass conservation pins the mean
    # current, so DC V/I says almost nothing about penetration.  What does
    # carry information is the pool surface oscillating under the arc: it
    # modulates the arc length, hence the voltage, at a frequency set by pool
    # size and an amplitude set by pool depth and superheat.
    C_osc: float = Field(
        3.80,
        description="lumped constant of f_osc = C * sqrt(gamma / (rho * (w/2)^3)) [-]",
    )
    f_osc_max: float = Field(400.0, description="clip on pool oscillation frequency [Hz]")
    k_a_osc: float = Field(
        0.060, description="oscillation amplitude per unit penetration [-]"
    )
    k_osc_T: float = Field(
        0.80, description="extra oscillation amplitude per unit pool superheat [-]"
    )

    # short-circuit statistics --------------------------------------------
    f_sc_max: float = Field(140.0, description="peak short-circuit frequency [Hz]")
    I_sc_peak: float = Field(150.0, description="current of peak short-circuit rate [A]")
    sigma_I_sc: float = Field(70.0, description="current spread of the sc-rate bell [A]")
    L_arc_ref: float = Field(5.0e-3, description="reference / commanded arc length [m]")
    k_sc_ratio: float = Field(
        0.058,
        description="sc-rate law: f_sc *= exp(-k * L_arc / a_osc); shorts need a big ripple [-]",
    )
    k_sag: float = Field(
        0.35,
        description="mean pool-surface depression per unit penetration [-]",
    )
    k_instab: float = Field(
        0.80, description="extra sc-rate jitter per unit pool superheat [-]",
    )
    # GMAW melting-rate (burn-off) law:  v_wire = a_burn*I + b_burn*stickout*I^2
    a_burn: float = Field(
        3.5e-4, description="burn-off law linear (arc) term [m/(s A)]"
    )
    b_burn: float = Field(
        4.3e-5, description="burn-off law resistive (stickout) term [1/(s A^2)]"
    )

    k_arc_force: float = Field(
        2.2e-5, description="arc force on the torch, F = k * I^2 [N/A^2]"
    )
    F_short: float = Field(1.5, description="extra torch force during a short circuit [N]")

    V_short: float = Field(6.0, description="voltage during a short-circuit event [V]")
    I_short_gain: float = Field(1.7, description="current surge factor during a short [-]")
    t_short: float = Field(1.8e-3, description="mean short-circuit event duration [s]")


class PowerSourceConfig(BaseModel):
    """Inverter power-source limits and inner-loop gains (ms timescale)."""

    I_min: float = Field(80.0, description="minimum weldable current [A]")
    I_max: float = Field(380.0, description="maximum current [A]")
    v_wire_min: float = Field(2.0 / 60.0, description="min wire feed speed [m/s]")
    v_wire_max: float = Field(14.0 / 60.0, description="max wire feed speed [m/s]")
    tau_I: float = Field(
        8.0e-3,
        description=(
            "current-loop time constant [s]; deliberately slower than the pool "
            "oscillation so the machine does not reject the very ripple we sense "
            "(this is what the machine's 'inductance' setting does)"
        ),
    )
    tau_wire: float = Field(60.0e-3, description="wire drive time constant [s]")
    L_arc_min: float = Field(1.0e-3, description="shortest physically sensible arc [m]")
    stickout_min: float = Field(3.0e-3, description="shortest electrode extension [m]")
    inner_dt: float = Field(
        2.0e-3, description="inner-loop update period [s]; ms timescale by design"
    )
    kp_arc: float = Field(
        0.45, description="inner-loop arc-length PI proportional gain [mm of command / mm of error]"
    )
    ki_arc: float = Field(
        1.20, description="inner-loop arc-length PI integral gain [1/s]"
    )
    kp_wire_I: float = Field(
        4.0e-5, description="current -> wire feed proportional gain [m/(s A)]"
    )
    ki_wire_I: float = Field(
        1.5e-4, description="current -> wire feed integral gain [m/(s A s)]"
    )


class RobotConfig(BaseModel):
    """Torch motion limits (10-100 ms timescale)."""

    v_travel_min: float = Field(2.0e-3, description="min travel speed [m/s]")
    v_travel_max: float = Field(14.0e-3, description="max travel speed [m/s]")
    a_travel_max: float = Field(40.0e-3, description="travel acceleration limit [m/s^2]")
    weave_amp_max: float = Field(4.0e-3, description="max weave half-amplitude [m]")
    weave_freq: float = Field(2.0, description="weave frequency [Hz]")
    tau_v: float = Field(50.0e-3, description="travel-speed servo time constant [s]")
    ctwd_nom: float = Field(
        20.0e-3, description="nominal contact-tip-to-work distance [m]"
    )
    torch_angle: float = Field(0.0, description="torch push/drag angle [rad]")


# --------------------------------------------------------------------------
# seam
# --------------------------------------------------------------------------
GapProfileKind = Literal["step", "ramp", "sine", "random", "constant"]


class SeamConfig(BaseModel):
    """Seam length and root-gap profile g(s)."""

    length: float = Field(0.20, description="seam length [m]")
    kind: GapProfileKind = "step"
    gap_min: float = Field(0.0, description="minimum root gap [m]")
    gap_max: float = Field(4.0e-3, description="maximum root gap [m]")
    smooth_len: float = Field(
        15.0e-3, description="Gaussian smoothing length applied to g(s) [m]"
    )
    noise_std: float = Field(0.10e-3, description="fit-up noise std on g(s) [m]")
    misalign_max: float = Field(
        1.0e-3, description="max lateral seam misalignment (offset) [m]"
    )
    ds: float = Field(0.5e-3, description="spatial resolution of the stored profile [m]")

    # --- plate thickness along the seam ------------------------------------
    # A stepped-thickness butt joint is an ordinary fabrication feature and a
    # good test of the planning layer: the step is on the drawing, it is not in
    # the gap scan, and the current that is right for the thick side burns
    # through the thin one.
    thickness_profile: Literal["uniform", "step_down"] = "uniform"
    thickness_step_at: float = Field(
        0.60, ge=0.0, le=1.0,
        description="fraction of the seam length at which the plate steps down [-]",
    )
    thickness_thin: float = Field(
        4.0e-3, gt=0.0, description="plate thickness after the step [m]"
    )


# --------------------------------------------------------------------------
# sensors (rates used from phase 2 onwards)
# --------------------------------------------------------------------------
class SensorConfig(BaseModel):
    """Rates and noise of the synthetic sensor suite."""

    f_master: float = Field(
        5000.0,
        description="master clock of the log [Hz]; channels faster than this are binned",
    )

    f_power: float = Field(5000.0, description="power-source V/I sample rate [Hz]")
    noise_V: float = Field(0.35, description="voltage noise std [V]")
    noise_I: float = Field(4.0, description="current noise std [A]")
    lsb_V: float = Field(0.01, description="voltage ADC resolution [V]")
    lsb_I: float = Field(0.25, description="current ADC resolution [A]")

    f_profiler: float = Field(30.0, description="laser seam profiler rate [Hz]")
    noise_gap: float = Field(0.12e-3, description="gap measurement noise std [m]")
    noise_offset: float = Field(0.15e-3, description="offset measurement noise std [m]")
    profiler_dropout: float = Field(0.06, description="P(dropout) per frame from spatter [-]")
    profiler_lead: float = Field(
        12.0e-3, description="profiler look-ahead distance ahead of the arc [m]"
    )

    f_ir: float = Field(30.0, description="IR camera rate [Hz]")
    noise_T: float = Field(25.0, description="IR peak-temperature noise std [K]")
    noise_w: float = Field(0.30e-3, description="IR isotherm pool-width noise std [m]")
    ir_smoke_tau: float = Field(
        0.35,
        description=(
            "IR radiance attenuation per unit smoke density [-]; applied to the "
            "DEVIATION from the calibration smoke level, because a real camera is "
            "calibrated on a nominal weld — so smoke costs you variance, not bias"
        ),
    )
    ir_width_atten_exp: float = Field(
        0.35,
        description=(
            "how much of the radiance attenuation reaches the pool-WIDTH reading "
            "[-]; the width comes from an isotherm gradient, the temperature from "
            "an absolute level, so the width is much less affected"
        ),
    )
    ir_blind_smoke: float = Field(
        1.70, description="smoke density above which the IR frame is rejected [-]"
    )

    f_mic: float = Field(20000.0, description="arc microphone rate [Hz]")
    noise_mic: float = Field(0.02, description="microphone broadband noise std [Pa]")

    f_force: float = Field(1000.0, description="torch force sensor rate [Hz]")
    noise_force: float = Field(0.05, description="force noise std [N]")

    f_mic_tone: float = Field(
        0.35, description="microphone tone amplitude at the pool oscillation [Pa]"
    )
    mic_broadband_per_kW: float = Field(
        0.045, description="broadband acoustic level per kW of arc power [Pa/kW]"
    )
    mic_click: float = Field(1.8, description="short-circuit re-ignition click amplitude [Pa]")

    f_rgb: float = Field(30.0, description="RGB camera rate [Hz]")
    rgb_smoke_tau: float = Field(
        3.2, description="RGB attenuation per unit smoke density [-] (high = blinded)"
    )
    rgb_glare_I: float = Field(
        260.0,
        description="current at which arc glare alone halves RGB usability [A]",
    )
    rgb_quality_min: float = Field(
        0.05, description="image quality below which the RGB frame is unusable [-]"
    )
    noise_rgb_w: float = Field(
        0.60e-3,
        description="RGB pool-width noise std at PERFECT visibility [m]; scaled by 1/quality",
    )
    smoke_base: float = Field(0.35, description="baseline smoke density [-]")
    smoke_ref: float = Field(
        0.88,
        description=(
            "smoke density the IR camera is CALIBRATED at [-]; set it to the "
            "nominal operating value (smoke_base + smoke_per_kW * nominal kW) so "
            "that fume costs the camera variance rather than a standing bias"
        ),
    )
    smoke_per_kW: float = Field(0.11, description="extra smoke density per kW of arc power [-]")


# --------------------------------------------------------------------------
# control
# --------------------------------------------------------------------------
class ControlConfig(BaseModel):
    """Motion-layer (10-100 ms) adaptive controller."""

    dt: float = Field(20.0e-3, description="motion-layer update period [s]")
    p_target: float = Field(4.0e-3, description="target penetration [m]")
    p_lo: float = Field(3.0e-3, description="lower edge of the acceptance band [m]")
    p_hi: float = Field(5.0e-3, description="upper edge of the acceptance band [m]")

    # --- handle allocation -------------------------------------------------
    # Current is the primary penetration authority (strong, direct).  Travel
    # speed is the productivity handle, bounded from above by the filler.
    # Weave bridges the gap and relieves over-penetration.
    kp_I: float = Field(1.10, description="penetration error -> current, proportional [-]")
    ki_I: float = Field(0.55, description="penetration error -> current, integral [1/s]")
    w_safety: float = Field(
        2.00,
        description="weight of the band-violation terms relative to target tracking [-]",
    )
    tau_integ: float = Field(
        1.50, description="integral bleed time constant when inside the band [s]"
    )
    kp_v: float = Field(0.9, description="penetration -> travel-speed proportional gain [-]")
    ki_v: float = Field(0.6, description="penetration -> travel-speed integral gain [1/s]")
    fill_target: float = Field(
        1.00, description="gap-fill ratio the travel-speed limit aims for [-]"
    )
    I_headroom: float = Field(
        0.15,
        description=(
            "fraction of the current range kept in reserve [-]; the speed loop "
            "stops pushing when the current loop is this close to saturating, "
            "because past that point speed can no longer be paid for"
        ),
    )
    k_prod: float = Field(
        0.35,
        description=(
            "productivity term [-]: when the whole +/-k_sigma band sits inside the "
            "acceptance band, speed up by this much per unit of remaining slack.  "
            "This is also what makes uncertainty costly -- a wide band leaves no "
            "slack, so an unsure controller simply does not speed up"
        ),
    )
    kp_wire: float = Field(0.7, description="fill error -> wire feed proportional gain [-]")
    k_weave_pen: float = Field(
        1.10, description="extra weave per unit of normalised over-penetration [-]"
    )
    I_min_cmd: float = Field(170.0, description="lowest current the motion layer may ask for [A]")
    I_max_cmd: float = Field(285.0, description="highest current the motion layer may ask for [A]")
    gap_lead_time: float = Field(
        0.50,
        description=(
            "how far ahead in time the controller acts on the previewed gap [s]; "
            "covers the travel-speed servo and the pool's own lag"
        ),
    )
    v_slew: float = Field(
        18.0e-3, description="max change of the travel-speed command [m/s per second]"
    )
    I_slew: float = Field(
        120.0,
        description=(
            "max rate of change of the current command [A/s].  Without this the "
            "PI chases the per-tick noise of the estimate and the machine hunts "
            "over tens of amps at 50 Hz -- correct on paper, unacceptable on a "
            "real inverter and audible in the arc"
        ),
    )
    k_sigma: float = Field(
        1.6,
        description="conservatism: extra margin per unit penetration std [-]",
    )
    sigma_ref: float = Field(
        0.30e-3, description="penetration std at which conservatism reaches k_sigma [m]"
    )
    weave_per_gap: float = Field(
        0.55, description="weave half-amplitude commanded per unit gap [-]"
    )
    bt_margin: float = Field(
        0.85, description="predicted-burn-through trip at p_pred > margin * thickness [-]"
    )
    bt_speed_boost: float = Field(1.45, description="emergency travel-speed multiplier [-]")
    bt_current_cut: float = Field(0.80, description="emergency current multiplier [-]")
    horizon: float = Field(0.25, description="burn-through prediction horizon [s]")


class BaselineConfig(BaseModel):
    """Fixed-parameter (non-adaptive) reference controller."""

    I_set: float = Field(230.0, description="fixed current setpoint [A]")
    v_travel: float = Field(4.5e-3, description="fixed travel speed [m/s]")
    v_wire: float = Field(6.88 / 60.0, description="fixed wire feed speed [m/s]")
    weave_amp: float = Field(0.0, description="fixed weave half-amplitude [m]")


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------
class EKFConfig(BaseModel):
    """EKF tuning.  Q/R are given as standard deviations, squared internally."""

    dt: float = Field(20.0e-3, description="filter update period [s]")
    q_T: float = Field(60.0, description="process noise std, pool temperature [K/sqrt(s)]")
    q_w: float = Field(1.60e-3, description="process noise std, pool width [m/sqrt(s)]")
    q_p: float = Field(1.20e-3, description="process noise std, penetration [m/sqrt(s)]")
    q_f: float = Field(0.15, description="process noise std, fill ratio [1/sqrt(s)]")

    r_fsc: float = Field(
        9.7,
        description=(
            "meas. noise std, short-circuit frequency [Hz]; IDENTIFIED FROM DATA and "
            "3x larger than the signal's own spread -- at this current the arc is "
            "globular, so this channel carries almost nothing.  Kept because it "
            "costs nothing and would matter in short-arc transfer"
        ),
    )
    r_power: float = Field(140.0, description="meas. noise std, mean arc power [W]")
    r_larc: float = Field(0.45e-3, description="meas. noise std, arc-length estimate [m]")
    r_ir_T: float = Field(45.0, description="meas. noise std, IR peak temperature [K]")
    r_ir_w: float = Field(0.5e-3, description="meas. noise std, IR pool width [m]")
    r_gap: float = Field(0.20e-3, description="meas. noise std, profiler gap [m]")
    r_rgb_w: float = Field(8.0e-3, description="meas. noise std, RGB pool width (bad) [m]")
    r_ripple_f: float = Field(
        2.95, description="meas. noise std, ripple frequency [Hz]; IDENTIFIED FROM DATA"
    )
    r_ripple_a: float = Field(
        0.0194e-3,
        description="meas. noise std, ripple amplitude as arc length [m]; IDENTIFIED FROM DATA",
    )
    k_ripple: float = Field(
        0.94,
        description=(
            "measured ripple amplitude / true surface amplitude [-]; the CV loop "
            "absorbs a few percent of the modulation.  IDENTIFIED FROM DATA "
            "(scripts/plot_estimation.py reports the value it measures)"
        ),
    )

    gap_std_profiler: float = Field(
        0.20e-3, description="uncertainty on the gap INPUT when the profiler works [m]"
    )
    gap_std_blind: float = Field(
        1.20e-3,
        description=(
            "uncertainty on the gap INPUT with no profiler [m] -- the fit-up "
            "tolerance the procedure has to assume"
        ),
    )
    gap_persistence: float = Field(
        1.0,
        description=(
            "correlation time of the gap-input error [s]; a wrong gap stays wrong "
            "for about this long, so one step's sensitivity is amplified by "
            "gap_persistence/dt to get the covariance it will actually accumulate"
        ),
    )

    p0_T: float = Field(80.0, description="initial std, pool temperature [K]")
    p0_w: float = Field(2.0e-3, description="initial std, pool width [m]")
    p0_p: float = Field(1.2e-3, description="initial std, penetration [m]")
    p0_f: float = Field(0.30, description="initial std, fill ratio [-]")


# --------------------------------------------------------------------------
# unmodelled disturbances (what makes the estimation problem non-trivial)
# --------------------------------------------------------------------------
class DisturbanceConfig(BaseModel):
    """Slow drifts the estimator does **not** know about.

    Without these the arc-length feature would recover penetration almost
    exactly and the multi-sensor fusion story would be vacuous.  Real cells
    have exactly these effects: contact-tip wear and stickout variation change
    the resistive voltage drop, and the arc root wanders.  Modelled as
    Ornstein-Uhlenbeck processes so they are smooth, zero-mean and seedable.
    """

    rho_e_tau: float = Field(6.0, description="contact-tip wear drift correlation time [s]")
    rho_e_std: float = Field(
        0.08, description="relative drift of the electrode-extension resistivity [-]"
    )
    ctwd_tau: float = Field(1.5, description="torch-height wander correlation time [s]")
    ctwd_std: float = Field(
        0.40e-3, description="torch-height (CTWD) wander std [m] — part/robot tolerance"
    )
    smoke_tau: float = Field(0.8, description="smoke density correlation time [s]")
    smoke_std: float = Field(0.18, description="smoke density fluctuation std [-]")


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------
class SimConfig(BaseModel):
    """Simulation clock."""

    f_sim: float = Field(5000.0, description="physics integration rate [Hz]")
    seed: int = 0
    t_max: float = Field(120.0, description="hard stop on simulated time [s]")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dt(self) -> float:
        """Physics timestep [s]."""
        return 1.0 / self.f_sim


class WeldConfig(BaseModel):
    """Root configuration object passed to every component."""

    #: V/I analysis window [s].  200 ms is a compromise: long enough for a 4 Hz
    #: frequency resolution on the pool ripple, short enough that the pool has
    #: not moved much within it (its fastest time constant is 20 ms, its
    #: gap-response constant ~100 ms).
    estimator_window: float = 0.20
    #: band searched for the pool-oscillation ripple [Hz]
    estimator_band: tuple[float, float] = (30.0, 400.0)
    #: Deliberate plant/model mismatch for the estimator, as multipliers on the
    #: lumped melt-pool coefficients.  Without this the estimator's model would
    #: BE the simulator's model, the filter would be unfairly good, and the
    #: learned residual would have nothing to learn.  These stand in for the
    #: parameter error you are left with after identifying a reduced-order model
    #: from a finite amount of real weld data.
    model_mismatch: dict[str, float] = {
        "eta_melt": 0.92,
        "C_cond": 1.12,
        "k_gap_ar": 0.88,
        "AR_0": 1.06,
        "kappa_L": 0.95,
    }

    material: MaterialConfig = Field(default_factory=MaterialConfig)
    consumable: ConsumableConfig = Field(default_factory=ConsumableConfig)
    joint: JointConfig = Field(default_factory=JointConfig)
    pool: PoolModelConfig = Field(default_factory=PoolModelConfig)
    defect: DefectConfig = Field(default_factory=DefectConfig)
    arc: ArcConfig = Field(default_factory=ArcConfig)
    power_source: PowerSourceConfig = Field(default_factory=PowerSourceConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    seam: SeamConfig = Field(default_factory=SeamConfig)
    sensors: SensorConfig = Field(default_factory=SensorConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    ekf: EKFConfig = Field(default_factory=EKFConfig)
    disturbance: DisturbanceConfig = Field(default_factory=DisturbanceConfig)
    sim: SimConfig = Field(default_factory=SimConfig)

    @model_validator(mode="after")
    def _check_bands(self) -> "WeldConfig":
        if not (self.control.p_lo < self.control.p_target < self.control.p_hi):
            raise ValueError("control: require p_lo < p_target < p_hi")
        if self.control.p_hi >= self.joint.thickness:
            raise ValueError("control: p_hi must be below plate thickness")
        if self.pool.AR_min >= self.pool.AR_max:
            raise ValueError("pool: AR_min must be < AR_max")
        return self


def estimator_config(cfg: "WeldConfig") -> "WeldConfig":
    """A copy of ``cfg`` with ``model_mismatch`` applied to the pool coefficients.

    This is the model the estimator and the controller are allowed to use.  The
    simulator keeps the unperturbed one.  Anything that evaluates the estimator
    against ground truth must build its process model through this function,
    or the result is meaningless.
    """
    out = WeldConfig.model_validate(cfg.model_dump())
    for field, factor in cfg.model_mismatch.items():
        setattr(out.pool, field, getattr(cfg.pool, field) * factor)
    return out


def default_config(**overrides) -> WeldConfig:
    """Build the nominal configuration, applying shallow ``section.field`` overrides.

    >>> cfg = default_config(seam__kind="ramp", sim__seed=3)
    """
    cfg = WeldConfig()
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        if not field:
            raise KeyError(f"override must be 'section__field', got {key!r}")
        setattr(getattr(cfg, section), field, value)
    return WeldConfig.model_validate(cfg.model_dump())
