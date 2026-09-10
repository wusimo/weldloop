# weldloop — 电源在环（Power-Source-in-the-Loop）自适应机器人焊接

> 纯仿真演示仓库，笔记本电脑上一条命令即可端到端运行，无需任何硬件。
> 所有模块都留有真实硬件适配器接口（`TODO(real-hw)`）。

---

## 一、论点（Thesis）

焊接过程中可见光视觉不可靠：烟尘、飞溅、弧光过曝会让 RGB 相机在最需要它的时刻失效。
**真正持续可用的过程传感器是焊接电源本身** —— kHz 级的电弧电压与电流。本仓库要证明的是：
熔池状态（熔深、熔宽、热输入、冷却速率）虽然不可直接观测，但可以用
"降阶熔池物理模型 + EKF + 小残差网络" 的方式在线估计并给出不确定度；再由一个
10–100 ms 的运动层控制器消费这个带协方差的估计，去调节行走速度、摆动幅度与送丝速度，
从而在 **0–4 mm 变坡口间隙** 的厚板 MIG/MAG 焊中把熔深稳定在目标带内，同时避免烧穿。

一个必须诚实说明的物理事实（本仓库据此建模）：
在恒压（CV）GMAW 中，**送丝速度通过熔化平衡钉死了平均电流，机器又钉死了平均电压**，
所以 V/I 的**直流量几乎不携带熔深信息**。真正携带信息的是熔池自由表面的振荡：
它调制瞬时弧长，从而在 kHz 电压流中留下可测的纹波——频率对应熔宽、幅值对应熔深。
`weldloop/physics/arc.py` 实现的正是这个机制，Phase 3 的特征提取器提取的也正是它。

---

## 二、三层控制架构（Architecture）

```
                                   ┌──────────────────────────────────────┐
   秒–分钟                          │  planning/task_planner.py            │
   task planning                    │  焊缝 → 分段 + 初始工艺窗口           │
   (NOT in the real-time loop)      │  TODO: LLM planner hook（离线）       │
                                   └───────────────┬──────────────────────┘
                                                   │ 工艺窗口 / 分段
                                                   ▼
   10–100 ms                       ┌──────────────────────────────────────┐
   torch-motion layer               │  control/motion_layer.py             │
   (本演示的重点)                    │  行走速度 / 摆幅 / 送丝设定值          │
                                   │  消费 EKF 的 (均值, 协方差)            │
                                   │  不确定度高 → 保守；预测烧穿 → 提速降流  │
                                   └──────┬────────────────────┬──────────┘
                                          │ 设定值              │ 估计
                                          ▼                    │
   ~ms                             ┌────────────────────┐      │
   power-source inner loop          │ control/inner_loop │      │
   (逆变电源本来就在做的事)          │ 电流波形/弧长 PI    │      │
                                   └──────┬─────────────┘      │
                                          ▼                    │
                                   ┌────────────────────┐      │
                                   │   sim/cell.py      │      │
                                   │   WeldCell 5 kHz   │      │
                                   │   电源自调节 + 电弧  │      │
                                   │   + 降阶熔池模型     │      │
                                   └──────┬─────────────┘      │
                                          │ 传感器             │
                                          ▼                    │
                                   ┌────────────────────┐      │
                                   │  sim/sensors.py    │      │
                                   │  V/I 5k · 激光 30  │      │
                                   │  IR 30 · 声 20k    │      │
                                   │  力 1k · RGB 30    │      │
                                   └──────┬─────────────┘      │
                                          ▼                    │
                                   ┌────────────────────┐      │
                                   │ estimation/ekf.py  │──────┘
                                   │ 物理先验 + 残差     │
                                   │ 输出 均值 + 协方差   │
                                   └────────────────────┘
```

---

## 三、当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 1 | `physics/` + `sim/`（热源、电弧、降阶熔池、焊缝、WeldCell） | ✅ 完成，70 tests |
| Phase 2 | `sim/sensors.py` + `sim/logger.py`，统一主时钟与宽表 schema | ✅ 完成，107 tests |
| Phase 3 | `estimation/`（V/I 特征 → EKF 融合 + 学习残差），RMSE 对比表 | ✅ 完成，136 tests |
| Phase 4 | `control/`（baseline vs adaptive vs **RGB 视觉**三方对比），指标对比表 | ✅ 完成，164 tests |
| Phase 5 | `viz/` + `scripts/run_demo.py`，出图与 metrics.json | ✅ 完成 |
| Phase 6 | 动画渲染：俯视对比 `out/weldloop.mp4` + 第三人称 `out/weldloop_robot.mp4` | ✅ 完成 |
| Phase 7 | **真实机器人在环**：MuJoCo + UR10e，Blender Cycles 照片级渲染 | ✅ 完成 |

> **关于"用 VLA 直接控制焊枪"**：本演示**不这样做**，理由是时间尺度与可观测性。
> 电源内环 ~1 ms、运动层 20 ms、任务规划 秒–分钟；熔深对间隙变化的响应时间常数
> ~100 ms，烧穿在 1 s 内发展完毕。VLA 最快也只有几 Hz，而且它的输入相机在
> Phase 2 中实测有 74 % 的帧不可用。把视觉放进实时回路，等于在视觉最不可靠的
> 工况下宣称视觉闭环——这与本方案的立论正好相反。
> AI 出现在两个诚实的位置：`estimation/residual.py`（估计器内的小残差网络）与
> `planning/task_planner.py`（秒级、离线的任务规划钩子，`TODO(llm-planner)` 标出了接入点：
> 把作业描述/WPS/装配扫描变成分段与初始工艺窗口，输出只是**起点**，下游本来就会偏离它）。
> Phase 4 会额外做一路 **RGB 视觉控制器** 作为对照组，用指标把"为什么不用视觉"
> 从断言变成实测结果。

运行：

```bash
cd weldloop
python -m pytest -q                      # 全部测试
python scripts/plot_physics.py           # Phase 1 自检图 -> out/
python scripts/plot_sensors.py           # Phase 2 传感器图 -> out/
python scripts/plot_estimation.py        # Phase 3 估计图 + RMSE 表 -> out/
python scripts/plot_control.py           # Phase 4 三方控制对比 -> out/
python scripts/run_demo.py --seed 0 --gap-profile step   # 主演示（约 28 s）
python scripts/render_animation.py       # 俯视对比动画 -> out/weldloop.mp4（约 2.5 min）
python scripts/render_robot.py           # 第三人称机器人动画 -> out/weldloop_robot.mp4

# Phase 7：真实机器人在环 + 照片级渲染（需要 pip install "weldloop[robot]" 与 Blender）
python scripts/render_mujoco.py          # MuJoCo 单元视频 -> out/weldloop_mujoco.mp4
python scripts/render_photoreal.py --stage all   # 全流程 -> out/weldloop_photoreal.mp4
python scripts/make_dataset.py --n 6     # 生成数据集 -> data/
python scripts/train_residual.py         # 可选：训练残差网络（无 torch 时自动跳过）
```

---

## 四、模型说明（Phase 1）

### `physics/heat_source.py`
经典解析热源，仅作为**结构参考与量级锚点**，不进入实时回路：

* **Rosenthal (1946)** 移动点热源准稳态温度场 → 熔合线等温线的解析宽度/深度；
* **Goldak (1984)** 双椭球体积热源 → 物理形状正确的功率密度分布。

按**名称**引用，**不引用其中任何参数数值**。所有几何半轴、效率均可配置，需用真实数据标定。

### `physics/melt_pool.py` — 降阶熔池模型
状态 `x = [T_pool, w, p, f]`（熔池温度、熔宽、熔深、间隙填充率），
输入 `u = [I, V, v_travel, v_wire, gap, thickness, weave_amp]`。
方程见该文件顶部 docstring（能量记账 → 熔化体积平衡 → 宽深比分配 → 过热度 → 填充率）。
稳态精确复现教科书熔化效率关系 `A = η_arc·η_melt·V·I / (ρ·h_m·v)`。
**烧穿**判据有两条：熔透板厚，或未填满时根部液态跨度超过表面张力桥接极限
`w_crit = c_st·sqrt(γ/(ρg))`——变间隙工况下先触发的是后者，这也符合实际。
**未熔合**判据：熔深不足 / 熔池宽度未润湿两侧坡口 / 填充不足。

> 该模型是**控制导向的降阶模型**，不是 CFD/有限元。所有系数都是集总标定常数。

### `physics/arc.py` — 电弧与"电源即传感器"
静态电弧特性 `V = V_0 + E_a·L_arc + R_stickout(I)·I`；GMAW 熔化（burn-off）律
`v_wire = a·I + b·stickout·I²`；熔滴过渡模式；熔池表面振荡
`f_osc = C_osc·sqrt(γ/(ρ(w/2)³))`、`a_osc = k_a·p·(1+k_T·过热度)`；短路频率
`f_sc ∝ exp(−k·L_arc/a_osc)`。

### `sim/cell.py` — WeldCell
5 kHz 定步长积分，内含**真实的 GMAW 自调节**：电极伸出长度是一个状态，
`d(stickout)/dt = v_wire − melting_rate(I, stickout)`，电流由 CV 特性在当前弧长下决定。
另含三个估计器不知道的慢漂移（导电嘴磨损、焊枪高度波动、烟尘密度），
这正是让"只用 V/I"的估计存在偏差、从而需要多传感器融合的原因。

`WeldCell.step()` 在给定 seed 下**逐位确定**。


---

## 五、传感器套件与采集 schema（Phase 2）

### 5.1 六个传感器，各自的失效模式

| 传感器 | 速率 | 它会在什么地方出错 |
|---|---|---|
| `PowerSourceSensor` | 5 kHz | 高斯噪声 + ADC 量化。**从不失效**——这正是本方案的立论点：烟尘、弧光、飞溅都打不掉它 |
| `SeamProfiler` | 30 Hz | 飞溅导致丢帧（丢帧报 NaN，绝不报一个"看起来合理的错数"）；**前视**安装，给的是预览而不是反馈 |
| `IRCamera` | 30 Hz | 烟尘衰减辐射。相机在额定烟尘下标定，所以烟尘带来的是**方差**而不是常值偏差；浓烟团时整帧拒绝 |
| `ArcMic` | 20 kHz | 宽带噪声 + 熔池振荡音调 + 再引弧冲击；比仿真步长还快，一步产生 4 个样本 |
| `TorchForce` | 1 kHz | 被电弧力主导，看不到熔池；只用于与短路统计互证 |
| `RGBCamera` | 30 Hz | **专门放进来演示它会瞎**：烟尘衰减系数远大于 IR，叠加弧光过曝 |

实测（`seed=0`，200 mm 阶跃间隙焊缝，由 `scripts/plot_sensors.py` 产生）：

* IR 熔宽 RMSE **0.83 mm**，RGB 熔宽 RMSE **7.97 mm**（熔池本身才 ~11 mm 宽）；
* RGB 有 **74 %** 的帧图像可用度低于阈值；即使"可用"的帧，其读数噪声也已达到熔宽量级；
* 激光轮廓仪领先电弧 **2.7 s** 看到间隙变化（12 mm 前视 / 4.5 mm·s⁻¹）；
* **规则：RGB 永远不作为过程传感器使用。** 它只出现在 Phase 3 RMSE 表的"反面例子"一行。

### 5.2 主时钟与宽表规则

1. **一个主时钟。** 每一行是同一个时钟的一拍（默认 5 kHz，即电源采样率）。传感器按**自己的时间戳**入表，而不是按软件何时读到。
2. **NaN 表示"该时刻没有采样"。** 30 Hz 相机每 167 行填一行。写文件时**不做前向填充**——插值是分析阶段的选择，把它烘进日志会毁掉"数据何时真正到达"这一证据。
3. **比主时钟更快的通道做块内聚合，而不是丢弃。** 20 kHz 麦克风每个主拍贡献一个 RMS 和一个咔哒计数；schema 的 `聚合` 列写明每个值的含义。真实产线上原始流单独存盘，块特征进主表——这里的 schema 就是照这个写的。
4. **真值被隔离。** 真实产线拿不到的列一律以 `truth_` 前缀标注、`real_hw=False`。估计器与控制器只能看到 `LogTable.real_hw_view()`；打分代码才读 `truth_` 列。有一条测试专门保证真值不会泄漏进控制器可见的 `Observation`。

`scripts/make_dataset.py` 按此 schema 生成数据集（Parquet，无 pyarrow 时退化为 gzip CSV），并附 `manifest.json` 与 `SCHEMA.md`。

### 5.3 采集 schema（= 一期真实采集建议表）

| column | unit | source | rate [Hz] | 主时钟聚合 | 真实产线可得 | 说明 |
|---|---|---|---|---|---|---|
| `t` | s | master clock | 5000 | last | ✅ | master clock time; one row per tick |
| `ps_V` | V | power source | 5000 | last | ✅ | arc voltage, raw (short-circuit collapses included) |
| `ps_I` | A | power source | 5000 | last | ✅ | welding current, raw (short-circuit surges included) |
| `ps_short` | - | power source | 5000 | max | ✅ | 1 while a short circuit is active |
| `ps_v_wire` | m/s | power source | 5000 | last | ✅ | wire feed speed from the drive tacho |
| `ps_V_set` | V | power source | 5000 | last | ✅ | machine set voltage (what the CV loop is holding) |
| `prof_gap` | m | laser profiler | 30 | last | ✅ | root gap measured AHEAD of the arc; NaN on spatter dropout |
| `prof_offset` | m | laser profiler | 30 | last | ✅ | lateral seam offset ahead of the arc |
| `prof_lead_s` | m | laser profiler | 30 | last | ✅ | seam station the profiler was looking at |
| `prof_valid` | - | laser profiler | 30 | last | ✅ | 0 on dropout; a dropout is never reported as a plausible number |
| `ir_T_peak` | K | IR camera | 30 | last | ✅ | peak apparent pool temperature; smoke-attenuated |
| `ir_pool_width` | m | IR camera | 30 | last | ✅ | pool width from the melting isotherm |
| `ir_valid` | - | IR camera | 30 | last | ✅ | 0 when the frame is rejected (dense plume) |
| `mic_p` | Pa | arc microphone | 20000 | rms | ✅ | RMS over the master tick of the 20 kHz acoustic pressure |
| `mic_click` | count | arc microphone | 20000 | sum | ✅ | short-circuit re-ignition clicks in this master tick |
| `force_N` | N | torch force | 1000 | last | ✅ | torch reaction force; dominated by arc force |
| `rgb_quality` | - | RGB camera | 30 | last | ✅ | image usability = smoke transmission x glare rejection |
| `rgb_pool_width` | m | RGB camera | 30 | last | ✅ | pool width from the visible image; noise scales as 1/quality |
| `rgb_valid` | - | RGB camera | 30 | last | ✅ | 0 when quality is below the usable threshold |
| `cmd_I_set` | A | controller | 5000 | last | ✅ | commanded current setpoint |
| `cmd_v_wire_set` | m/s | controller | 5000 | last | ✅ | commanded wire feed speed |
| `cmd_arc_len_set` | m | controller | 5000 | last | ✅ | commanded arc length |
| `cmd_v_travel` | m/s | controller | 5000 | last | ✅ | commanded travel speed |
| `cmd_weave_amp` | m | controller | 5000 | last | ✅ | commanded weave half-amplitude |
| `rb_s` | m | robot encoder | 5000 | last | ✅ | torch position along the seam |
| `rb_v_travel` | m/s | robot encoder | 5000 | last | ✅ | actual travel speed |
| `rb_weave_offset` | m | robot encoder | 5000 | last | ✅ | instantaneous lateral weave offset |
| `rb_ctwd` | m | robot | 5000 | last | ✅ | commanded contact-tip-to-work distance |
| `truth_gap` | m | simulator | 5000 | last | ❌ 仅仿真 | true root gap under the arc |
| `truth_offset` | m | simulator | 5000 | last | ❌ 仅仿真 | true lateral misalignment |
| `truth_T_pool` | K | simulator | 5000 | last | ❌ 仅仿真 | true mean pool temperature |
| `truth_pool_w` | m | simulator | 5000 | last | ❌ 仅仿真 | true pool width |
| `truth_penetration` | m | simulator | 5000 | last | ❌ 仅仿真 | true penetration depth — the quantity being estimated |
| `truth_fill` | - | simulator | 5000 | last | ❌ 仅仿真 | true gap fill ratio |
| `truth_stickout` | m | simulator | 5000 | last | ❌ 仅仿真 | true electrode extension |
| `truth_arc_len` | m | simulator | 5000 | last | ❌ 仅仿真 | true mean arc length including pool depression |
| `truth_f_osc` | Hz | simulator | 5000 | last | ❌ 仅仿真 | true pool oscillation frequency |
| `truth_a_osc` | m | simulator | 5000 | last | ❌ 仅仿真 | true pool oscillation amplitude |
| `truth_f_sc` | Hz | simulator | 5000 | last | ❌ 仅仿真 | true expected short-circuit rate |
| `truth_smoke` | - | simulator | 5000 | last | ❌ 仅仿真 | true smoke density |
| `truth_burn_through` | - | simulator | 5000 | max | ❌ 仅仿真 | 1 while the burn-through condition holds |
| `truth_lack_of_fusion` | - | simulator | 5000 | max | ❌ 仅仿真 | 1 while any lack-of-fusion condition holds |


---

## 六、状态估计（Phase 3）

### 6.1 先说一个不方便但必须说的事实

原始设想是"平均电弧功率和短路频率就能跟上熔深"。**实测：不行。**
在恒压 GMAW 中，送丝速度通过熔化平衡钉死平均电流，机器钉死平均电压，所以直流量几乎不动。
下表是 seed=0、200 mm 阶跃间隙焊缝上，每个 V/I 特征与真实熔深的相关系数（由代码实测）：

| V/I 特征 | 与真实熔深的 \|相关系数\| |
|---|---|
| 纹波频率 `f_ripple` | **0.98** |
| 纹波幅值 `a_ripple` | **0.90** |
| 短路频率 `f_sc` | 0.40 |
| 平均电流 `I` | 0.29 |
| 平均电弧功率 `V·I` | 0.29 |
| 弧长估计 `L_arc_est` | 0.26 |
| 平均电压 `V` | 0.01 |

信息在**波形结构**里，不在直流量里。物理机制是熔池自由表面振荡调制瞬时弧长：

* `f_ripple = C_osc·sqrt(γ/(ρ(w/2)³))` → **熔宽**，实测增益 0.999、残差 RMS 2.95 Hz；
* `a_ripple = k·a_osc(p, 过热度)` → **熔深**，实测增益 0.940、残差 RMS 0.019 mm。

> 注意：`f_ripple` 直接测的是熔宽，它与熔深的高相关来自本工况下"间隙张开 → 熔池变窄变深"的耦合。
> 这一步转换由 EKF 的物理模型完成，而不是把频率当熔深用。
> 另外 `f_sc` 的实测噪声（9.7 Hz）比信号本身的变化幅度（3.1 Hz）还大 —— 在 230 A 熔滴过渡下它几乎不带信息。
> 保留它是因为代价为零，且在短路过渡工况下它会变得重要。

### 6.2 EKF 设计要点

* **过程模型故意与被控对象不一致**。估计器跑的是 `config.estimator_config()`
  ——把降阶模型系数扰动 5–12 % 后的副本，代表"用有限真实数据辨识后残留的参数误差"。
  否则估计器就是在用仿真器自己的模型解自己的题，结果没有意义，残差网络也无事可学。
* **测量协方差 R 由数据辨识**，不是拍脑袋给的（见 6.1 的残差 RMS）。
* **滑窗特征的相关性被显式处理**：V/I 特征窗长 200 ms、步长 20 ms，相邻窗共享 90 % 的样本，
  把它们当独立测量会让协方差按 √N 假性收缩——这正是"自信地错"的成因。R 按重叠倍数放大。
* **坡口间隙是模型输入，不是测量**。轮廓仪不观测熔池；间隙错了，新息无法纠正，只会污染预测。
  因此间隙的不确定度被显式传播进 P（`gap_std_blind` 1.2 mm / `gap_std_profiler` 0.2 mm）。
  这也是"仅 V/I"与"V/I + 轮廓仪"差距的来源。
* **输出协方差供控制器消费**，Phase 4 的运动层据此变保守。

### 6.3 RMSE 对比表（seed=0，200 mm 阶跃间隙焊缝，全部由本仓库代码运行产生）

| 传感器组合 | 熔深 RMSE [mm] | 熔深偏差 [mm] | 熔宽 RMSE [mm] | 滤波器自报 σ_p [mm] | 2σ 覆盖率 |
|---|---|---|---|---|---|
| 仅电源 V/I | 0.446 | −0.384 | 0.234 | 0.380 | 0.93 |
| V/I + 激光轮廓仪 | 0.380 | −0.350 | 0.192 | 0.306 | 0.95 |
| 全部（V/I + 轮廓仪 + IR） | 0.336 | −0.308 | 0.204 | 0.305 | 0.98 |
| **仅 RGB 相机（反面例子）** | **1.388** | −1.240 | 1.686 | **1.231** | 0.98 |
| 全部 + 学习残差 | **0.127** | −0.036 | 0.212 | 0.321 | 1.00 |

读法：

1. **只用电源就已经可用**：0.45 mm RMSE，无相机、无轮廓仪。这是本方案的核心主张。
2. 加轮廓仪与 IR 逐步收敛到 0.34 mm，且**协方差单调收缩**（σ_w：0.69 → 0.44 → 0.36 mm），
   即滤波器确实"知道自己知道得更多了"。
3. **RGB 是反面例子**：误差 1.39 mm（熔深本身才 3–5.5 mm），但滤波器自报 σ 1.23 mm ——
   它诚实地报告"我不知道"，而不是自信地错。这正是为什么 RGB 不进实时回路。
4. **学习残差**把误差降到 0.13 mm、偏差降到 −0.04 mm。它修正的是**动力学**而非测量，
   即那部分"辨识不准"的模型误差。残差在 seeds 100–102 的 step/ramp/sine 焊缝上训练，
   在完全未见的焊缝上验证：

   | 未见焊缝 | 全部传感器 | + 学习残差 |
   |---|---|---|
   | step, seed 0 | 0.336 mm | 0.127 mm |
   | random, seed 7 | 0.438 mm | 0.243 mm |
   | sine, seed 11 | 0.407 mm | 0.186 mm |
   | ramp, seed 13 | 0.304 mm | 0.100 mm |

   无 torch 时 `load_residual()` 返回 `None`，EKF 退化为纯物理，README 两种结果都给出，
   学习部分从不是必需项。
5. 一个诚实的不足：加了残差后误差 0.13 mm 而 σ 仍 0.32 mm，滤波器变**偏保守**——
   因为 Q 是按无残差模型调的。生产版本应在带残差的条件下重新整定 Q。


---

## 七、控制器对比（Phase 4）

### 7.1 三层时间尺度是真的分开跑的

| 层 | 周期 | 代码 | 它在做什么 |
|---|---|---|---|
| 原始 V/I 流 | 0.2 ms | `AdaptiveController.on_samples` | 把 5 kHz 波形推进特征环形缓冲（真实产线上是 DAQ 回调） |
| 电源内环 | 2 ms | `control/inner_loop.py` | 用送丝速度给电流、用设定电压给弧长，各一个慢 PI |
| 运动层 | 20 ms | `control/motion_layer.py` | 提特征 → EKF → 决定行走速度/摆幅/电流 |

`inner_loop.py` **只做逆变电源本来就没做的那一点点**：CV 机器已经自己稳弧长、
自己靠熔化平衡调节伸出长度（10 ms 量级），再加一个跟它抢方向盘的控制器只会更糟。

### 7.2 手柄分配

| 手柄 | 角色 | 依据 |
|---|---|---|
| **电流**（经送丝） | 熔深的主要权限 | 直接、强；PI 跟踪 `p_target`，band 违规项权重是跟踪项的 `w_safety`=2 倍 |
| **行走速度** | 生产率手柄 | 只在"整个 ±k·σ 区间都在验收带内"**且**"电流环还有余量"时才提速；上限由填充能力硬卡 |
| **摆幅** | 桥接间隙 / 润湿两侧 / 缓解过熔深 | 由前视间隙前馈 + 熔宽下置信界驱动 |

**不确定度是怎么变贵的**：提速量正比于置信区间到验收带边缘的 *余量*。
区间越宽 → 余量越小 → 不提速。保守不是额外打的补丁，是从这一条里自然掉出来的。
硬安全另走一路：用过程模型把当前估计外推 `horizon`=0.25 s，若预测熔深（含 σ 裕度）
触到 `bt_margin`×板厚，立即提速 + 降流 + 满摆，完全绕开 PI。

电流指令带 `I_slew`=120 A/s 限幅。没有它，PI 会去追估计量每一拍的噪声，
机器在 50 Hz 上摆几十安培——纸面没问题，真机上不可接受，而且电弧声就能听出来。

### 7.3 指标对比（seed=0，200 mm 阶跃间隙 0–4 mm，6 mm 板）

| 控制器 | 烧穿孔洞 | 烧穿长度 | 未熔合 | 熔深 std | 在带内 | 无缺陷长度 | 平均速度 | 循环时间 |
|---|---|---|---|---|---|---|---|---|
| 定参数 baseline | **10** | **35.6 mm** | 1.2 mm | 0.60 mm | 73.4 % | 82.2 % | 4.50 mm/s | 44.4 s |
| RGB 视觉 vision | 3 | 3.8 mm | 4.5 mm | 0.47 mm | 96.6 % | 95.9 % | 2.69 mm/s | **74.4 s** |
| **自适应 adaptive** | **0** | **0.0 mm** | **0.0 mm** | **0.18 mm** | **100 %** | **100 %** | 4.33 mm/s | 46.2 s |
| 自适应 + 学习残差 | **0** | **0.0 mm** | **0.0 mm** | 0.19 mm | **100 %** | **100 %** | 4.23 mm/s | 47.3 s |

"孔洞"只统计持续长度 ≥ `min_hole_length`(0.2 mm) 的连续段——毫秒级的判据抖动不是板上的洞，
按采样点数去数会得到一个随采样率变化的假指标。

### 7.4 五条焊缝上的稳健性（step×2 / ramp / random / sine，150 mm）

单条焊缝的漂亮结果说明不了什么，所以把三方对比在 5 条不同 seed、不同间隙形状的焊缝上重跑：

| 控制器 | 平均孔洞数 | 平均烧穿长度 | 平均未熔合 | 平均熔深 std | 平均在带内 | 平均循环时间 |
|---|---|---|---|---|---|---|
| 定参数 baseline | 4.0 | 7.9 mm | 0.0 mm | 0.45 mm | 79.0 % | 33.3 s |
| **自适应 adaptive** | **0.0** | **0.1 mm** | 0.1 mm | **0.19 mm** | **99.2 %** | **33.3 s** |
| RGB 视觉 vision | 0.0 | 0.3 mm | 2.4 mm | 0.38 mm | 99.1 % | 54.8 s |

**自适应把烧穿长度降到 1/79、熔深标准差减半、在带内比例从 79 % 提到 99 %，而循环时间一模一样。**

### 7.5 关于 RGB 视觉这一路，结论要说准确

它不是"把焊缝焊废了"。它做出的焊缝基本可接受——**代价是慢了 65 %**。
机理很清楚：它的熔深估计偏差 1.2 mm、自报 σ 也是 1.2 mm，于是 7.2 节里的提速条件
（"整个置信区间都在带内"）几乎永远不成立，它只能一路爬行；而且它频繁触发烧穿预警。
所以正确的说法是：

> 把视觉放进实时回路，不是会立刻出事，而是**你要用 65 % 的节拍去买它的不确定度**，
> 并且换来的仍然是更差的熔深控制（std 0.38 mm vs 0.19 mm）与 24 倍的未熔合长度。

对照实验是干净的：**两路用的是同一套控制律、同一个物理先验 EKF、同样的协方差处理，
唯一区别是哪些测量进了滤波器。** 差距只能归到传感器。
（这一路也**不是** VLA/学习策略，本仓库不做那种声称；但同样的问题会原样传给任何策略类：
Phase 2 实测电弧燃烧期间约 74 % 的帧不可用，剩下的帧熔宽误差与熔池本身同量级。
换个策略类救不了一个被遮蔽的传感器。）

### 7.6 调了什么参数（按 Phase 4 的要求说明）

初版自适应控制器**输给了** baseline：循环时间 97.7 s（+120 %），且烧穿更多。三处修正：

1. **发现一个真 bug**：控制器根本没拿到间隙。`rb_s`/`rb_v_travel` 是日志列，不是传感器通道，
   `Observation` 里压根没有机器人反馈，所以前视预览一直读到 0。
   已在 `Observation.update_machine()` 中补上（只写真实产线可得的机器侧回读，不碰任何 `truth_`）。
2. **重新分配手柄**。初版用行走速度控熔深、送丝控填充，结果两者互相打架：零间隙时填充前馈把电流
   压到下限，熔深塌了，速度环又把速度压到下限去补，节拍翻倍。改为电流控熔深、速度控生产率
   （填充作为速度的单边上限）。
3. **安全监视器不许比自己的信念更乐观**：`p_pred = max(模型外推, p_hat)`。
   模型已知有偏，滤波器正靠测量把状态顶住，外推值系统性偏低。

另外整定了：`kp_I`=1.10 / `ki_I`=0.55（电流环）、`k_prod`=0.35（生产率项）、
`I_slew`=120 A/s（指令限幅，对指标不敏感，纯为真机可用性）。


---

## 八、一条命令跑完整个演示（Phase 5）

```bash
python scripts/run_demo.py --seed 0 --gap-profile step
```

在笔记本 CPU 上约 **28 s** 跑完，产生：

| 文件 | 内容 |
|---|---|
| `out/summary.png` | **给幻灯片用的单图**：上=间隙曲线，中=真实熔深（定参数 vs 自适应）+ 估计 ±2σ + 验收带，下=原始 5 kHz V/I 与短路事件 |
| `out/dashboard.png` | 工程视图：两个控制器四行并排（扰动/缺陷/指令/在线估计） |
| `out/metrics.json` | README 里每一个数字 |

可选参数：`--vision` 加上 RGB 视觉对照组，`--gap-profile {step,ramp,sine,random,constant}`，
`--no-residual` 强制纯物理。给定 `--seed` 后结果逐位可复现。

`summary.png` 底部那一栏是整个方案的问题陈述：**定参数焊接时间隙从 0 走到 4 mm，
而电压 26.6 ± 0.02 V、电流 223 ± 3.2 A —— 机器仪表上几乎什么也没发生。**
信息在波形结构里，这正是 Phase 3 要提取的东西。

---

## 九、动画渲染（Phase 6）

### 9.1 俯视对比 `scripts/render_animation.py`

```bash
python scripts/render_animation.py --seed 0 --gap-profile step
```

产出 `out/weldloop.mp4`（约 2.5 min 渲染，20 s 视频）。画面分三块：

* **俯视**：Rosenthal 解析温度场、坡口两侧边线、身后凝固的焊道、焊枪（含摆动）、
  以及从电弧脱落并向后飘散的烟团；
* **横截面**：按真实比例画出母材、坡口、熔池半椭圆、余高与填充，烧穿时根部标红；
* **相机**：RGB 与 IR 并排 —— RGB 被烟尘与弧光糊成一片灰噪声并标出"不可用"，
  IR 仍能画出熔合等温线。这就是 Phase 2 那两个 RMSE 数字的画面版。

两个控制器**按位置同步播放**（而不是按时间），所以同一横坐标上比较的是同一段焊缝；
每一路各自带一个时钟，循环时间的差别直接看得见。

> **必须说清楚**：这是**同一套降阶物理的可视化**，不是 CFD。温度场是解析解、
> 熔池是降阶模型的半椭圆、烟羽是程序化烟团。图上写了这句话，因为一张"看起来像 CFD
> 但不是"的图在提案里是负资产。


### 9.2 第三人称机器人视角 `scripts/render_robot.py`

```bash
python scripts/render_robot.py --seed 0 --gap-profile step
```

产出 `out/weldloop_robot.mp4`：一台机械臂端着焊枪沿焊缝走，母材按解析温度场发亮，
焊道在枪后凝固变暗，烟尘从电弧升起；右侧同步给出 RGB / IR 相机画面与熔池横截面，
底部是两种控制方式的真实熔深对比，并在"定参数在此处已烧穿"的位置弹出提示。

**哪些是真的、哪些是布景（必须分清）**

| 内容 | 来源 |
|---|---|
| 焊枪位置、行走速度、摆动、电流电压、熔宽熔深、烟尘密度、缺陷标志 | **仿真日志**，与其余所有图表同源 |
| 母材表面温度场 | **Rosenthal 解析解**，用当前瞬时功率与行走速度算出 |
| 关节角 | 对上述 TCP 轨迹做**逆运动学**，并校核关节限位 |
| 连杆长度、焊枪外形、烟团与火花 | **布景**。不参与任何物理计算——把机械臂整个删掉，焊缝一模一样 |

`weldloop/viz/robot.py` 是一台**通用小型弧焊臂**（0.64 m 臂展，肘型 + 球型腕）的运动学，
连杆参数可配置、不取自任何厂商样本。它不只是画着好看：

* 正/逆运动学互为逆映射，测试断言 TCP 位姿往返误差 < 1e-11；
* 逆解在工作空间外**抛异常**而不是返回一个看似合理的错值；
* `solve_path()` 对整条焊缝求解并报告"是否可达、最差关节余量、腕部是否接近奇异"。
  这正是运动学模型平常的用途——**建线之前先确认这条轨迹机器人走得了**。

一个实际发现：最初为了取景把机器人放得离工件很近，结果**肘关节只剩 6° 余量**；
按"最大化最差关节余量"重新选底座位置后变成 **39°**。这条搜索就在模块的 docstring 里。
另一个：`atan2` 在 ±π 处回绕，会让腕部滚转在焊缝中段跳 2π、渲染出来机械臂会"抽一下"；
`solve_path()` 现在做 `np.unwrap`，并有一条测试专门盯住它。


---

## 十、真实机器人在环与照片级渲染（Phase 7）

前六个阶段把机械臂当作一个"行走速度伺服"，因为焊接物理只需要焊枪的位置与速度。
这站得住脚，但留下两个客户一定会问的问题：**真机走得了这套运动吗？跟踪误差会把焊缝弄成什么样？**

### 10.1 把 UR10e 放进回路

`weldloop/sim/mujoco_cell.py` 用 MuJoCo 搭了一个真实焊接单元：
**MuJoCo Menagerie 原版 UR10e**（真实连杆惯量、关节限位、随模型发布的 PD 位置执行器）、
焊枪、工作台、夹具与工件。它实现的是**同一个 `RobotBase` 接口**——这正是那个抽象基类存在的意义：

```python
robot = MujocoRobot(cfg, seam)              # 一台真的 6 轴机械臂
simulate(cfg, controller=..., robot=robot)  # 其余一行都不用改
```

* 指令路径参数仍由控制器的行走速度积分而来（与 `SimRobot` 同一套律，比较才公平）；
* 阻尼最小二乘微分逆解把 TCP 位姿变成关节目标；
* MuJoCo 用自己的执行器动力学积分整条手臂；
* **随后把实际达到的 TCP 反投影回焊缝，交给焊接物理**。

于是跟踪误差、伺服滞后与摆动衰减会真的传到熔池里。

### 10.2 结论在真机上是否成立

| | 行走速度伺服 | **UR10e 在环** |
|---|---|---|
| 定参数：烧穿长度 | 35.6 mm | **36.3 mm** |
| 自适应：烧穿长度 | 0.0 mm | **0.0 mm** |
| 自适应：熔深 std | 0.19 mm | **0.18 mm** |
| 自适应：在带内 | 100 % | **100 %** |
| 循环时间 | 47.3 s | 48.1 s（+1.7 %） |
| TCP 跟踪误差 | — | **0.006–0.056 mm** |

**结论原样成立。** 这不是"运气好"，而是因为 2 Hz、±2 mm 的摆动对一台工业臂来说本来就不难；
把它算出来，比在提案里写一句"机器人应该跟得上"有用得多。

### 10.3 一路上真发现的两个问题

1. **Menagerie 的伺服增益与摆动共振。** 随模型发布的 `kp=5000, kd=500` 的慢极点在
   `kp/kd = 10 rad/s ≈ 1.6 Hz`，正好压在 2 Hz 摆动上，实测摆幅被放大到指令的 **1.9 倍**。
   这是通用抓取整定的性质，不是硬件的性质；按焊接工况重新整定为 `kp=14000, kd=220`
   后，TCP 平均误差 **0.134 mm**、速度跟踪精确。这两个数是本仓库里唯一与已发布模型不同的地方，
   代码里写明了。
2. **逆解闭在测量上会留下静差。** 把 `q_cmd = qpos + gain·dq` 这样闭环，平衡点只要求
   `gain·dq` 等于伺服下垂量，于是稳态 TCP 误差恒为 **11 mm**。改成在**指令**上积分
   （resolved-rate + 前馈）后降到 0.13 mm。另外给机械臂开了重力补偿——真实控制器都这么做，
   否则那 1 厘米下垂会被误读成一个控制结果。

### 10.4 渲染

两条渲染管线，同一份数据：

* `scripts/render_mujoco.py` → **MuJoCo 渲染器**：真实网格、阴影、随焊枪推进逐段点亮并冷却的焊道、
  作为**真实光源**的电弧（它照亮工件并投出影子）。快，几分钟出片。
* `scripts/render_photoreal.py` → **Blender Cycles**：UR10e 原始网格、金属材质、
  作为体积介质的**焊接烟尘**（电弧的光会在里面散射）、随温度由白热冷却成暗色焊道的着色器。
  两个机位：单元全景与焊枪特写。

分工是严格的：**weldloop 负责物理，MuJoCo 负责运动学与臂动力学，Blender 只负责像素。**
画面上烧进去的每一个数字都来自仿真日志；渲染不参与任何计算。

> 渲染过程中修掉的几个 bug 值得记一笔，它们都是"看上去对、其实错"的那类：
> Blender 的 `primitive_cube_add(size=1.0)` 跨度是 ±0.5，所以 `scale` 给的是**全长**——
> 我按半长给，焊道只画了半条焊缝，着色器的位置映射也差了一倍；
> AgX 色调映射会把过亮的自发光**去饱和成白色**，焊道发光强度调到 220 才在正确的曝光下呈现橙色；
> 焊枪一开始是按"从 TCP 往回量"摆的，而 MJCF 是从法兰量的，于是它飘在半空。
> 这些都是渲一帧看一眼才发现的，不是靠读代码。

---

## 十一、参数来源声明

仓库中所有数值只有两类：

1. **低碳钢的教科书物性**（密度、比热、导热、熔点、熔化潜热），两位有效数字；
2. **降阶模型的集总标定常数**，没有独立物理含义，取值使本演示的额定工作点落在
   合理的焊道几何范围内。

**没有任何一个数值来自某篇文献的具体报道**，全部可配置，全部需要用一期真实采集数据重新辨识。
README 中所有指标数字都由本仓库代码实际运行产生。

---

## 十二、走向真实硬件时需要替换的适配器

| 抽象基类 | 仿真实现 | 真实硬件适配器（待写） |
|---|---|---|
| `interfaces.PowerSourceBase` | `sim.cell.SimPowerSource` | 逆变电源现场总线 / SDK |
| `interfaces.RobotBase` | `sim.cell.SimRobot` 或 `sim.mujoco_cell.MujocoRobot`（UR10e） | 机器人 EGM / RSI 运动流。MuJoCo 版已经是"半只脚踏进真实"：把 `mj_step` 换成 EGM 流、把测量 TCP 换成机器人自身反馈即可，上层不动 |
| `viz.robot.ArmGeometry` | 通用小型弧焊臂 | 换成实机连杆参数与关节限位，即可用同一套 `solve_path()` 做建线可达性校核 |
| `interfaces.SensorBase` | `sim.sensors.*` | 每个物理传感器一个 |
| `sim.seam.make_seam` | 合成间隙曲线 | 激光轮廓仪实测 / 装配扫描 |
| `physics.melt_pool` 系数 | 集总默认值 | 用一期数据做参数辨识 |
| `estimation.residual` | 在仿真数据上训练 | 在真实数据上重训 |
