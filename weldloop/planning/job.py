"""The job the planning demo is built around, as a job *card* and a *scan*.

Two objects, split by how the information arrives:

``JobCard``
    Free text.  Drawing notes, the WPS window, the acceptance rule, and the
    shop-floor remarks that never make it into any structured field.  This is
    the half a rule-based planner structurally cannot use — not because rules
    are stupid, but because every fact in here would need somebody to add a
    field for it first.
``FitupScan``
    An image.  The laser profiler's report as the inspector exports it.  The
    planner is handed the picture, not the array, so "it re-cut the seam in
    the right places" is a claim about reading a plot rather than about
    numpy.

Both are deliberately *documents*.  The point of putting a language model at
this layer is that documents are what a welding shop already has.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from weldloop.config import WeldConfig, default_config
from weldloop.planning.task_planner import SeamDescription
from weldloop.sim.seam import Seam, make_seam

__all__ = [
    "JobCard", "FitupScan", "DEMO_JOB",
    "demo_config", "demo_seam", "demo_seam_description",
]


@dataclass(frozen=True, slots=True)
class JobCard:
    """The paperwork that comes with a job."""

    job_id: str
    scan_date: str
    text: str
    #: the acceptance rule, as a fraction of plate thickness
    p_frac_lo: float
    p_frac_hi: float


@dataclass(frozen=True, slots=True)
class FitupScan:
    """The inspection report image."""

    path: Path
    caption: str = "坡口装配检测报告 / joint fit-up inspection report"

    def exists(self) -> bool:
        return self.path.is_file()


# --------------------------------------------------------------------------
# the demo job
# --------------------------------------------------------------------------
#: acceptance rule.  These two fractions are where ``control.p_lo`` and
#: ``control.p_hi`` come from on 6 mm plate, so the job card and the config
#: cannot drift apart: 3.0 mm / 6.0 mm = 0.50, 5.0 mm / 6.0 mm = 0.833.
_P_FRAC_LO, _P_FRAC_HI = 0.50, 0.833

_JOB_TEXT = """\
工件号 JOB            WL-2409-017
母材 MATERIAL         Q235B 低碳钢 / mild steel
接头 JOINT            对接, I 型坡口(不开坡口), 单道单面焊
焊缝长度 LENGTH       200 mm
位置 POSITION         PA 平焊

板厚 PLATE THICKNESS
    前段 6 mm; 自焊缝 120 mm 处起为 4 mm（见图纸 A-A 剖视）。
    台阶是机加工出来的, 不是装配误差, 激光扫描测的是根部间隙, 测不到板厚。

根部间隙 ROOT GAP
    按装配实测, 见附检测报告。名义 0 mm, 装配公差 4 mm。

焊丝/保护气 CONSUMABLES
    ER70S-6, φ1.2 mm; 80% Ar + 20% CO2, 15 L/min。

工艺窗口 PROCESS WINDOW (机器与 WPS 允许范围)
    焊接电流      {I_lo:.0f} – {I_hi:.0f} A
    焊接速度      {v_lo:.1f} – {v_hi:.1f} mm/s
    摆动半幅      0 – {a_hi:.1f} mm

验收 ACCEPTANCE
    熔深 ≥ 板厚的 {f_lo:.0%}, ≤ 板厚的 {f_hi:.1%}; 不允许烧穿, 不允许未熔合。
    注意这是按板厚的比例写的, 不是一个固定的毫米数。

车间备注 SHOP NOTES
    这批板是激光切割后组对的, 中段对不齐, 间隙偏大。
    上一批同型号工件在薄板段烧穿过两件, 返修成本很高, 薄板段请偏保守。
"""


def _job_text(cfg: WeldConfig) -> str:
    return _JOB_TEXT.format(
        I_lo=cfg.control.I_min_cmd,
        I_hi=cfg.control.I_max_cmd,
        v_lo=cfg.robot.v_travel_min * 1e3,
        v_hi=cfg.robot.v_travel_max * 1e3,
        a_hi=cfg.robot.weave_amp_max * 1e3,
        f_lo=_P_FRAC_LO,
        f_hi=_P_FRAC_HI,
    )


def demo_config(seed: int = 0) -> WeldConfig:
    """The planning demo's cell: the standard config on a stepped plate.

    Everything except ``seam.thickness_profile`` is the demo's own default, so
    any difference from the headline run is attributable to the joint, not to
    a quietly retuned controller.
    """
    return default_config(
        seam__kind="step",
        seam__length=0.20,
        seam__thickness_profile="step_down",
        seam__thickness_step_at=0.60,
        seam__thickness_thin=4.0e-3,
        sim__seed=seed,
    )


def demo_seam(cfg: WeldConfig | None = None, seed: int = 0) -> Seam:
    cfg = cfg or demo_config(seed)
    return make_seam(cfg.seam, seed=seed, thickness=cfg.joint.thickness)


def demo_seam_description(cfg: WeldConfig | None = None) -> SeamDescription:
    """What a *structured* planner interface can hold about this job.

    Note what is missing, and note that nothing here is a lie: there is one
    ``thickness`` field, and 6 mm is the honest answer to "how thick is this
    job".  The step exists only in ``notes``, which is exactly where such
    things live in practice.
    """
    cfg = cfg or demo_config()
    return SeamDescription(
        length=cfg.seam.length,
        thickness=cfg.joint.thickness,
        joint_type="square_butt",
        material="mild_steel",
        position="PA",
        gap_nominal=0.0,
        gap_tolerance=cfg.seam.gap_max,
        notes=_job_text(cfg),
    )


DEMO_JOB = JobCard(
    job_id="WL-2409-017",
    scan_date="2026-09-14",
    text=_job_text(default_config()),
    p_frac_lo=_P_FRAC_LO,
    p_frac_hi=_P_FRAC_HI,
)
