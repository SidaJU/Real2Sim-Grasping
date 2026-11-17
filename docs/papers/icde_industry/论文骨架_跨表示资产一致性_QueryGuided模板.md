# 论文骨架：跨表示三维仿真资产的数据一致性管理

- 日期：2026-11-17
- 目标赛道：ICDE 2027 Industry and Applications（截止 2026-12-01 23:59 PT）
- 模板来源：Query-Guided Analysis and Mitigation of Data Verification Errors（ICDE 2026，原文在 `refs_icde_quality/`）
- 状态：**执行计划与脚本设计；无实验结果，不承诺录用**。
- 前提（GO/NO-GO）：本机当前无 `.npy`/`.ckpt`/`results/`。**拿不到数据就停工**；不要靠编造结果推进。

---

## 0. 一句话论文

> 三维仿真资产由多阶段异构产物组成（RGB/相机 → 3DGS → 深度 → mesh → 碰撞体 → USD）。单个文件合法不代表整套产物彼此一致。我们把"跨表示不一致"形式化为数据管理问题：用**资产依赖图**追踪检查错误如何传播到交付结果，提出**最坏情况交付风险**指标，并在有限预算下**引导最值得做的检查与最小修复**。

与 DRE（单 GLB 可用性）、CoMe（几何重建）的区别写进 related work，不重复。

---

## 1. 必须写进论文的形式化（照 Query-Guided 的骨架）

### 1.1 资产依赖图（对应它的 provenance）

定义有向无环图 \(G=(V,E)\)：
- 节点 \(v\in V\)：一个产物（obs/rgb、camera、depth_gs、mesh、collider、usd…）或一次检查
- 边 \((u,v)\in E\)：u 是 v 的输入/引用来源
- 节点属性：`artifact_id`, `type`, `version`, `content_hash`, `check_state ∈ {pass, fail, unknown}`, `confidence ∈ [0,1]`, `cost`

**边要表达三种依赖**：生成（depth→mesh）、引用（usd→mesh）、约定（camera+depth→mesh 的坐标/尺度）。第三种是你的创新点，别人没有。

### 1.2 检查错误传播（对应它的 label error propagation）

规则：上游 `fail/unknown` 沿边传播到下游交付节点 `usd_delivery`。写清：
- **必然传播**：尺度漏转（R3）→ USD 尺寸错；USD 错引（R4）→ 加载错资产
- **不传播**：某视角遮挡但另有视角覆盖 → 不影响最终 mesh 的某局部（要写清条件）
- **检查本身可能出错**：L0 通过不代表几何真实（false negative），所以要有 confidence

### 1.3 最坏情况交付风险（对应 MES）

给定当前检查状态与置信度，定义：
\[
\mathrm{WR}(o) = \min_{\text{额外检查集合 } S} \; \Pr_{\text{worst}}[\text{交付节点 } o \text{ 违反任务要求} \mid \text{已做检查}, S]
\]
简化实现（赶工期版，不必 ILP）：**worst-case = 所有 unknown 节点按最坏赋值时，是否存在一条从 fail/unknown 到交付节点的依赖路径**；用可达性 + 规则命中计数做启发式分数。论文里明确这是 worst-case 的保守估计，不是概率保证。

### 1.4 选择问题（对应 MESReduce）

目标：在预算 B（检查时间 + 人工分钟）内，选下一组检查/修复动作 \(S\)，最大化 \(\mathrm{WR}\) 下降（或交付可靠性提升）。
- 输出：`继续交付 / 追加检查 L1/L2 / 定位到阶段 X / 拒绝`
- 基线对比：Random、Probability-Greedy（按置信度低先查）、Structure-Greedy（按依赖度先查）、固定全检——对应它的 4 个 baseline。

---

## 2. 脚本清单：复用什么、新写什么

所有新脚本放 `scripts/icde_pipeline/`，输入/输出统一 JSONL（一行一条记录），**不覆盖已有成功脚本**。

### 2.1 复用（只读扩展，不改原逻辑）

| 已有脚本 | 复用它做什么 | 改造 |
|---|---|---|
| `scripts/22_render_gs_depth.py` | 渲染 depth/rgb/acc npy | 不改；在输出目录旁写一条 `artifact` 记录 |
| `scripts/23b_gs_to_mesh_v2.py` | TSDF 融合出 mesh | 把 `MASK_ERODE/ACC_MIN/VOXEL/SKIP_LOW_RING` 写进记录 |
| `scripts/23c_compare_meshes.py` | Chamfer/边界边/bbox | 输出改成同时 print + JSONL |
| `scripts/23d_coverage_check.py` | 4000 点 × 63 帧覆盖 | 修幸存者偏差：同时检查过滤前候选点/掩码/任务区 |
| `scripts/17_setup_grasp_asset.py` | 生成 USD / 碰撞体 | **必须记录实际加载的是 gs_mesh 还是 gt_mesh** |
| `scripts/grasp/19d_grasp_side.py` 等 | 抓取试验 | 记录 (资产版本, 种子, 步数, SUCCESS/FAIL) |

### 2.2 新写（按开发顺序）

| # | 脚本 | 输入 | 输出 | 干什么 |
|---|---|---|---|---|
| S1 | `record_schema.py` | — | `schema.json` | 定义 JSONL 字段（见 §2.3），所有脚本共用 |
| S2 | `wrap_run.py` | 已有脚本的 stdout | `runs.jsonl` | 包装 22/23b/23c/23d/17/19，把 stdout + 配置 + 时间戳 + 产物 hash 落成结构化记录（数据泵） |
| S3 | `build_lineage.py` | `runs.jsonl` | `lineage.json` | 构建依赖图（生成/引用/约定三类边），输出邻接表 |
| S4 | `inject_faults.py` | 正常产物副本 | `faults_manifest.jsonl` | 受控注入：尺度漏转、坐标翻转、USD 错引、掩码过度、GT/重建资产混用；位置与强度预先固定 |
| S5 | `check_L0.py` | 产物 + schema | `checks_L0.jsonl` | 元数据/格式：身份、观测配对、单位、变换、引用版本、空/退化几何 |
| S6 | `check_L1.py` | 深度/mesh/碰撞 | `checks_L1.jsonl` | 跨表示规则 R1–R6：深度回投影、轮廓覆盖、支撑/接触区、mesh vs 碰撞差异 |
| S7 | `check_L2.py` | USD + Isaac | `checks_L2.jsonl` | 导入/静置/碰撞/有限动作回放（可选，最贵） |
| S8 | `worst_risk.py` | `lineage.json` + checks | `risk.jsonl` | 算每个交付节点的 WR 与风险路径（启发式可达性 + 规则命中） |
| S9 | `select_next.py` | risk + 预算 | `decisions.jsonl` | 选下一步检查/修复/拒绝，输出决策理由（可解释） |
| S10 | `repair_ops.py` | 失败规则 | `repairs.jsonl` | 有限修复算子：重新变换导出 / 重新绑定 USD / 冻结 GS 重融合 / 拒绝；只重做受影响阶段 |
| S11 | `evaluate.py` | 全部 jsonl + 人工标签 | `metrics.json` + 4 张图 | 检出率/误拒率/漏放/交付率/成本/人工分钟；AUC of reliability-vs-cost |
| S12 | `make_tables.py` | `metrics.json` | 论文表格 | 对照表、消融表、成本拆分表 |

### 2.3 JSONL 字段（S1 定义，最小集）

```json
{
  "run_id": "obj003_cfg07_seed2",
  "object_id": "003_cracker_box",
  "artifact_type": "mesh|depth|usd|collider|check|decision|repair",
  "artifact_id": "sha256 或自增",
  "version": "v2",
  "created_at": "ISO-8601",
  "upstream_ids": ["..."],
  "source": "gs|gt|synthetic",
  "length_unit": "m|normalized",
  "coord_convention": "OpenGL|OpenCV",
  "normalize_scale": 0.001,
  "camera_count": 99,
  "config": {"MASK_ERODE":8,"ACC_MIN":0.10,"VOXEL":0.002},
  "code_version": "git hash",
  "check_rule": "R1|R2|...|L0|L1|L2",
  "check_state": "pass|fail|unknown",
  "confidence": 0.0,
  "cost_ms": 0,
  "human_minutes": 0,
  "decision": "deliver|inspect|repair|reject",
  "repair_op": "retransform|rebind|refuse|none",
  "failure_type": "scale|camera|ref|mask|occlusion|none"
}
```

---

## 3. 数据从哪来（三个渠道，按优先级）

**渠道 A（首选）：旧机器/外置盘拷回**
- 旧机器路径：`C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb\`（99 PNG + transforms.json + points3D.ply）、`...\outputs\ycb\splatfacto\2026-09-23_102733\step-000029999.ckpt`、`...\exports\ycb\gs_depth\`（99 depth + 99 rgb + 99 acc）
- `C:\isaacsim_data\`（GT mesh/USD）
- 拷回后放 `data/`（`.gitignore` 已排除，不入库），先跑一遍 22→23b 确认能复现历史 Chamfer 3.00mm
- **拿不到 → 渠道 B**

**渠道 B：用 YCB 原始资产 + Isaac 重新采集**
- 源资产：Isaac 云端 `/Isaac/Props/YCB/Axis_Aligned/*.usd`（21 个对象）
- 采集：`scripts/06_capture_multiring.py`（3 高度 × 36 视角 = 108 帧 RGB-D + 内参）
- 初始化：`07_build_init.py` → 训练 CoMe/splatfacto（NVIDIA GPU，Ubuntu/CUDA 更稳；CoMe submodule 已 pin `caf62f6`，需 `git submodule update --init`）
- 先 3–5 个对象（饼干盒 + 钻头 + 圆柱 + 凹形 + 薄结构），确认方向后再扩到 8–12

**渠道 C：从已有产物派生（只作 proof-of-concept，不能当工业证据）**
- 视角子集（99 → 63/36/18）造不同观测证据
- 受控故障注入（S4）：尺度漏转、坐标翻转、USD 错引、掩码过度、GT/重建混用
- **论文必须写清：自然故障 vs 注入故障分开统计**（对应 Query-Guided 的 real-labels / worst-case / average-case 三场景）

**不解决的硬伤**：真实工业使用证据、多对象规模。这两项拿不到，论文只能写"研究原型"，不能写"生产部署"。

---

## 4. 实验设计（照 Query-Guided 的三场景 + 四基线）

### 4.1 数据划分（对象级，严禁泄漏）
- 首轮：3–5 对象，其中 2 开发、1 校准、1–2 测试（若只有 3 个，就全作开发，结论收窄）
- 扩轮：8–12 对象，按 5 开发 / 2 校准 / 3–5 测试（示例）
- **同一物体的不同视角/故障变体不得跨划分**——像素不是独立样本

### 4.2 三场景（对应它的 RLBL/WCS/AVG）
| 场景 | 定义 |
|---|---|
| RLBL 自然故障 | 你 TODO.md/HTML 调试报告里真实发生过的错误，重建可重放 |
| WCS 最坏情况 | 所有检查 confidence=0.499、状态默认 pass（模拟"看起来都过但都不可信"） |
| AVG 平均情况 | 随机注入故障，强度从预设集合均匀采样 |

### 4.3 方法对照（共享输入、冻结模型、算子、缓存、预算）
| 组 | 配置 | 排除什么 |
|---|---|---|
| B0 | 现有流程 + 人工检查（你的真实踩坑史） | 建立实际基线 |
| B1 | 固定全检（L0+L1+L2 全跑）+ 普通缓存 | 排除"收益只是多做了检查" |
| B2 | 通用贪心/搜索（同样规则，按置信度或依赖度排序，固定顺序修复） | 排除"收益只是多了修复算子" |
| Q | 本方法（依赖图 + WR + 分级 + 定位 + 最小重算） | 验证增量 |
| Oracle | 用测试 GT 选最优检查集合（**仅离线上界**） | 看还剩多少空间，绝不做在线输入 |

### 4.4 消融（对应它的不同 baseline）
- Q−graph：去掉依赖图，只做文件内检查
- Q−worst：去掉 WR，按固定阈值检查
- Q−locate：保留检查，修复改固定顺序全重跑
- Q−adaptive：检查计划固定为"最常用的一套"

### 4.5 指标
**主指标（预先冻结，不能事后改）**：
- 坏资产检出率（自然故障单独报）
- 好资产误拒率
- 最终漏放风险（放行资产中仍不达标比例；无放行标 N/A，不写 0）
- 交付率
- **可靠性-成本曲线下面积**（横轴：检查+修复+人工总分钟；纵轴：1−漏放风险 或 F1）
- 修复成功率（可修复故障中最终达标比例）

**次指标**：各阶段耗时、人工活跃分钟数、缓存命中、runtime（毫秒级）、每个对象的 bbox/Chamfer（几何只作辅助，不作主主张）

---

## 5. 预期结果（写成继续/停止信号，不是承诺）

| 信号 | 继续的条件 | 停止/收窄条件 |
|---|---|---|
| 问题存在（RQ1） | 自然管线里能重放 ≥2 类跨表示错误，且 B1 固定检查漏掉其中至少 1 类 | 所有错误都能被普通格式检查抓到，或 3 天找不到可重放自然故障 |
| 指标有用（RQ2） | WR 在独立测试对象上能把高风险资产排在低风险前面（如风险最高 20% 中命中多数坏资产） | WR 与真实质量不相关，或只在开发对象有效 |
| 决策有用（RQ3） | 同漏放下 Q 比 B1/B2 总成本低约 20–30%（**这是目标不是结果**）；或同预算交付率更高 | 收益 < 检查/图构建开销；Q 不显著胜过 B2 |
| 组件有用（RQ4） | 去掉 graph/worst/locate 后性能下降，至少两个消融有明确差异 | 消融无差异 → 新机制只是包装 |
| 应用有效（RQ5） | 资产准备的人工分钟数下降，且仿真导入/静置可用率不降 | 只有仿真视频、无重复试验和独立参考 → 收窄为"诊断工具" |

**绝不写**：没有数据就写百分比；用像素数当样本量；把"全部拒绝"说成低风险成功；把 GT 辅助结果说成 RGB-only；把仿真交互说成真机 sim-to-real。

---

## 6. 四张图 + 六页结构（论文成品清单）

| 图/表 | 内容 | 数据来源 |
|---|---|---|
| 图1 系统图 | 资产流 + 依赖图 + L0/L1/L2 + 定位/修复路径 | 你的设计 |
| 图2 失败案例 | 同一资产修复前后 + 独立证据（含修不好的例子） | S4 注入 + 自然故障 |
| 图3 可靠性-成本曲线 | Q/B0/B1/B2/Oracle 在 RLBL/WCS/AVG 的曲线 | S11 |
| 表1 对照与消融 | 检出/误拒/漏放/交付/成本/人工分钟 | S12 |
| 表2 成本拆分 | 采集/训练（共同）/检查/修复/仿真验证/人工 | runs.jsonl |

六页预算：问题与动机 1 页、数据模型与形式化 1 页、机制 1.5 页、实验 2 页、相关/局限 0.5 页。

---

## 7. 14 天时间表（今天 11/17 → 12/01 截止）

| 日期 | 工作 | GO/NO-GO |
|---|---|---|
| 11/17–18 | 确认渠道 A/B 数据可获取；写 S1/S2；跑通 1 个对象的结构化记录 | 拿不到数据 → 停工 |
| 11/19 | S3 依赖图 + S4 故障注入；跑 3–5 对象 | 找不到自然故障 → 收窄或停 |
| 11/20 | S5/S6 检查器 + S8 WR | WR 与质量完全不相关 → 停 |
| 11/21 | S9 决策 + S10 修复算子；跑 B0/B1/B2/Q | Q 不胜 B2 → 停止主线 |
| 11/22–23 | S11/S12 + 四张图；补消融 | 数字没出来 → 不硬写 |
| 11/24–27 | 写六页稿；补相关工作和边界 | — |
| 11/28–30 | 复跑关键结果、作者审阅、合规检查 | 数据/许可问题 → 不投 |
| 12/01 | 提交（23:59 PT，北京时间 12/02 15:59） | — |

写作从 D1 的问题提纲开始，不等实验结束才构思。

---

## 8. 最大风险（必须写进论文局限，不能藏）

1. 单对象/少对象 → 结论只能收窄为"研究原型"
2. 无真实部署 → 不能写生产收益，只能写流程证据
3. CoMe 主线是 Marching Tetrahedra，你的方法只在 DTU/单对象 TSDF 线 → 明确范围
4. 依赖图和 WR 是启发式，不是形式化保证（赶工期版）→ 写清假设
5. 故障注入不能替代自然故障 → 两类分开统计
6. 审稿人可能问"这不就是数据校验+缓存" → 用 B1/B2 和消融回答

---

## 9. 证据边界

本文件是执行计划，所有脚本、指标、预期均为**待实现/待验证**，不是成果；形式化是赶工期简化版；数据可获取性与环境可用性未确认。模板论文 Query-Guided 的原文在 `refs_icde_quality/`，可逐句复核。
