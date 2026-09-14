"""The prompt and the output schema for the R4 task-planning layer.

Kept in its own module for three reasons.  It is the part a process engineer
should be able to read and argue with; it is hashed so a recorded response can
be matched to the exact request that produced it; and it is the only place in
the repository where the phrasing of a question to a model lives.

What the prompt is allowed to ask for is narrow on purpose: a segment table
and a process window.  It is not asked for a control law, it is not asked for
setpoints "during" the weld, and it is told outright that the numbers it
returns are starting points that a 50 Hz loop will move away from.
"""

from __future__ import annotations

import hashlib
import json

from weldloop.config import WeldConfig

__all__ = ["PLAN_SCHEMA", "SYSTEM_PROMPT", "build_user_text", "request_digest"]


SYSTEM_PROMPT = """\
你是焊接工艺工程师,负责**离线**编制一条焊缝的分段工艺卡。

你的输出会作为机器人焊接单元的**起始工艺参数**。单元内部有一个 50 Hz 的自适应
运动层和一个熔池状态估计器,它们会在起弧之后根据实时电信号继续调整电流、速度和
摆动。所以:

- 你给的不是最终参数,是**每一段的出发点**。段内的间隙变化由下层负责。
- 你**不**参与实时控制。你的延迟是秒级,熔池对间隙变化的响应时间常数是 0.1 s
  量级,烧穿在 1 s 内就会发生。不要试图描述"焊到某处时再改"这类时序动作。
- 你能做而下层做不到的事只有一件:**读懂图纸、工艺卡和检测报告里的信息,把一条
  焊缝切成若干段,并给每段一个合适的出发点和验收带**。

分段原则:
- 段边界应该落在**工况真正发生变化的位置**,比如间隙明显变宽/变窄的拐点、板厚
  台阶。不要机械地均分。
- 段数取工况实际需要的数量,通常 3–6 段。段太多没有意义,因为下层本来就在连续调节。
- 每一段的 thickness_mm 必须填这一段**实际的板厚**。它会直接进入烧穿保护的判据,
  填错比不填更危险。
- 验收带 p_lo_mm / p_hi_mm 按工艺卡里的**比例规则**乘以该段板厚算出来,不要照抄
  厚板段的毫米数。

所有参数必须落在工艺卡给出的机器窗口内。超出窗口的数值会被单元夹到边界并记录为
工艺偏差,那对谁都没好处。
"""


#: The structured-output schema.  Every field carries its unit in the name:
#: the single most common way a plan goes wrong is metres against millimetres,
#: and a name is cheaper than a validator.
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reading": {
            "type": "string",
            "description": (
                "先用两三句说明你从检测报告图和工艺卡里读到了什么: 间隙沿焊缝怎么变化, "
                "拐点大概在哪些位置, 板厚在哪里变化。"
            ),
        },
        "segments": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "s_start_mm": {"type": "number"},
                    "s_end_mm": {"type": "number"},
                    "gap_class": {"type": "string", "enum": ["tight", "nominal", "wide"]},
                    "gap_mm": {
                        "type": "number",
                        "description": "这一段有代表性的根部间隙,从图上读",
                    },
                    "thickness_mm": {
                        "type": "number",
                        "description": "这一段的实际板厚",
                    },
                    "I_set_A": {"type": "number"},
                    "v_travel_mm_s": {"type": "number"},
                    "weave_amp_mm": {"type": "number", "description": "摆动半幅"},
                    "p_target_mm": {"type": "number"},
                    "p_lo_mm": {"type": "number"},
                    "p_hi_mm": {"type": "number"},
                    "rationale": {
                        "type": "string",
                        "description": "一句话说明这一段为什么是这些参数",
                    },
                },
                "required": [
                    "s_start_mm", "s_end_mm", "gap_class", "gap_mm", "thickness_mm",
                    "I_set_A", "v_travel_mm_s", "weave_amp_mm",
                    "p_target_mm", "p_lo_mm", "p_hi_mm", "rationale",
                ],
                "additionalProperties": False,
            },
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "你这份工艺卡建立在哪些假设上,以及哪些你其实不知道",
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "这条焊缝最可能出问题的地方,以及下层应该盯住什么",
        },
    },
    "required": ["reading", "segments", "assumptions", "risks"],
    "additionalProperties": False,
}


def build_user_text(job_text: str, cfg: WeldConfig, seam_length: float) -> str:
    """The text half of the request.  The image is attached alongside it."""
    return f"""\
下面是一条焊缝的工艺卡, 附图是这条焊缝的坡口装配检测报告(激光扫描 + 图纸剖视)。

工艺卡里给的是名义值和窗口, **实际的根部间隙只能从附图上读**——文字里没有间隙
数据, 扫描仪也测不到板厚。

--- 工艺卡 JOB CARD ---
{job_text}
--- 工艺卡结束 ---

焊缝全长 {seam_length * 1e3:.0f} mm, 从 s = 0 到 s = {seam_length * 1e3:.0f} mm。
单元的运动层更新周期 {cfg.control.dt * 1e3:.0f} ms; 摆动频率 {cfg.robot.weave_freq:.1f} Hz。

请给出分段工艺卡。
"""


def request_digest(job_text: str, cfg: WeldConfig, seam_length: float,
                   image_bytes: bytes | None) -> str:
    """A stable fingerprint of one planning request.

    A recorded response is only replayed for the request that produced it.
    Anything that would change what a model sees — the prompt, the schema, the
    job card, the machine limits, the scan image — changes this digest, and a
    stale recording then refuses to be used instead of quietly lying.
    """
    h = hashlib.sha256()
    h.update(SYSTEM_PROMPT.encode())
    h.update(json.dumps(PLAN_SCHEMA, sort_keys=True, ensure_ascii=False).encode())
    h.update(build_user_text(job_text, cfg, seam_length).encode())
    if image_bytes is not None:
        h.update(hashlib.sha256(image_bytes).digest())
    return h.hexdigest()[:16]
