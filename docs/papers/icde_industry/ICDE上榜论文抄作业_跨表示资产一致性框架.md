# ICDE 真实上榜论文"抄作业"：跨表示资产一致性管理框架

- 整理日期：2026-11-17
- 目的：从 ICDE 2024–2026 **真实录用**的数据质量/验证类论文里提炼可复用套路，套到你的"跨格式三维仿真资产数据一致性管理"新问题上。
- 本轮下载并**读了原文**的三篇（PDF + 文本在 `refs_icde_quality/`）：
  1. **Query-Guided Analysis and Mitigation of Data Verification Errors**（ICDE 2026, DOI 10.1109/ICDE65706.2026.00151；arXiv:2603.08612v2，45 页）
  2. **From Benchmarks to Production: Transferring Time Series Anomaly Detection Methods for Electricity Production Monitoring**（ICDE 2026；arXiv:2609.39257，13 页；EDF/Inria）
  3. **RAMSeS: Robust and Adaptive Model Selection for Time-Series Anomaly Detection Algorithms**（ICDE 2026；arXiv:2602.21766，15 页）
- 已知但未取得开放 PDF（仅摘要/官方页核实，**不作为原文依据**）：CBAClean（ICDE 2025 demo）、SOUND（ICDE 2025）、CDMap（ICDE 2025 industry）、GaV（ICDE 2026）。
- 阅读规则：可借用套路标明来自哪篇、原文关键句；你的对应设计标【本项目】；不把别人的贡献写成你的。

---

## 0. 三篇上榜论文的共同套路（一句话版）

| 论文 | 一句话套路 | 为什么能中 ICDE |
|---|---|---|
| **Query-Guided Data Verification**（ICDE 2026） | 已有的数据验证工具会出错 → 不替换它，而是**量化错误如何传播到下游查询结果** → 给最坏情况指标 MES + 指导额外验证以优化成本/可靠性 | 不发明验证器，做"验证的验证 + 成本决策"——纯数据管理问题 |
| **TAMIS / From Benchmarks to Production**（ICDE 2026） | 把学术界时序异常检测搬到 EDF 真实生产 → 按工业约束（CPU、可解释、专家 in-the-loop）重设架构 → 日报 top-ranked 异常给专家 | 真实工业痛点 + 系统 + 开放数据 + accuracy/efficiency tradeoff |
| **RAMSeS**（ICDE 2026） | 没有通用最优异常检测器 → 做**自适应的检测器选择**（LinTS + ε-greedy + GA 集成）→ 无标签也能鲁棒选择 | "没有 one-size-fits-all"是数据系统问题，不是调参 |

**共同点**：没有一篇在发明新模型；全都是"**已有的工具不够用/用不好 → 用一个数据管理机制把它用好，并证明成本收益**"。这正是你的模板。

---

## 1. 可借用的具体机制（逐篇拆）

### 1.1 Query-Guided Data Verification → 你最该抄的一篇

**原文关键句**（我读了 PDF 原文）：
> "verification can still produce erroneous labels that propagate to downstream query results in complex ways."

> "verification can be costly in terms of latency and resources... resulting labels may still be uncertain or erroneous, especially under cost-accuracy tradeoffs."

> "a notable shortcoming of existing systems is that the potential error propagation is not quantified and therefore not reported properly to the analysts."

**它的机制**：
1. **来源追踪（provenance）**：用布尔 k-DNF 公式表示"哪些输入元组决定哪个输出元组"——这是它的"血缘"
2. **最坏情况指标 MES**：不靠数据分布假设，量化每个输出元组的最坏错误
3. **反直觉指标 risky tuples**：减少某些输入的标签不确定性，反而可能增加输出不确定性——这是它的"新洞察"
4. **MESReduce**：不替换验证器，而是**引导外部验证器去验证最值得验证的输入**，在预算内优化可靠性
5. **实验**：真实标签（NELL 150 万元组）+ 合成（TPC-H）；三种场景（real-labels / worst-case / average-case）；4 个 baseline（Random / Formula-Count Greedy / Occurrences-Count Greedy / Probability Greedy）；指标是 **F1 vs verification cost 曲线**、MES 的对数比、runtime（<3ms 多数查询）

**【本项目】怎么抄**：
| 它的元素 | 你的对应 |
|---|---|
| 输入元组的 correctness label + error probability | 资产产物的 `check_result` + 检查置信度（未通过/通过但可疑） |
| 查询 provenance（k-DNF） | 资产依赖图：哪个 mesh 来自哪组深度/视角/参数，哪个 USD 引用哪个 mesh |
| MES（最坏情况错误） | **资产最坏可用性分数**：当前证据下，这个资产最坏可能错在哪（尺度/引用/碰撞）而未被检出 |
| risky tuples（减少不确定性反而更糟） | **关键洞察点**：某些"看起来通过"的检查（比如 USD 能加载、mesh 非空）反而会掩盖跨表示不一致——给专家验证指向这里 |
| MESReduce 引导外部 verifier | 你的分级检查：L0 全跑、L1 只对可疑资产、**L2 仿真回放只对高风险资产**——在预算内最大化交付可靠性 |
| 不替换已有工具，complement | 你不替换 CoMe/现有管线，只做它之上的一致性层——审稿友好 |

**这一篇贡献了你论文里最值钱的一句话**：
> 我们不是又做一个验证器，而是量化已有检查/验证的**错误如何在资产依赖图中传播**，并指导有限的额外检查资源投向最值得检查的产物。

### 1.2 TAMIS / EDF → 抄"工业系统论文"的壳

**原文关键句**：
> "practical deployment of anomaly detection systems in industry remains difficult due to specific constraints that often arise in operational settings."

> "designed for human-in-the-loop workflows... Only the most abnormal cases are returned, allowing experts to review them and make decisions."

**工业约束（原文列了 C1–C4）**：near-real-time、standard CPU no GPU、interpretability、data quality robustness（缺失值等）。

**【本项目】怎么抄**：
- 把你的系统写成**生产约束驱动**：不假设无限 GPU、专家（你）时间有限 → L0/L1 便宜检查全跑、**只把 top-k 可疑资产交给人工/仿真验证**（对应它的 daily newsletter + expert review）
- 贡献列表照它的写法分条：Novel Industrial Architecture / Lightweight method / Open data / Superior trade-off
- 开放数据：你至少公开结构化记录 schema、故障注入脚本、标注的坏资产案例（哪怕只有几个）

### 1.3 RAMSeS → 抄"没有万能方法"的选型问题

**原文关键句**：
> "Time-series data vary widely across domains, making a universal anomaly detector impractical."

**【本项目】怎么抄**：
- 你的问题里也可以说："不同用途/不同资产，没有万能的检查计划"——这就是你 SimAsset-Q 文档里"不同任务需要不同计划"的 ICDE 化版本
- 把"选检查计划"写成一个**数据驱动的适应性选择**：LinTS/ε-greedy 选下一步检查哪条规则、 Thompson sampling 选修复算子——但这要时间，赶工期可先做规则版，把 adaptive 作为 future work

---

## 2. 你的论文框架（直接可套的结构）

**题目（候选）**：
> *Consistency-Aware Delivery of Cross-Representation Simulation Assets: Quantifying and Mitigating Propagated Verification Errors*

**摘要套路**（照 Query-Guided 的句式）：
> 三维仿真资产由多阶段异构产物构成... 现有检查工具（格式验证、mesh 检查、人工验收）可能给出错误/不完整的判定，这些错误会沿依赖图传播到 USD 与仿真任务，导致静默不可用。...我们不替换现有工具，而是量化跨表示验证错误如何传播到交付结果，提出 [你的指标]，并引导有限的额外检查/修复预算...

**贡献列表**（三条，照 EDF 的写法）：
1. **问题形式化**：首次把"跨表示资产一致性"形式化为数据管理问题，给出依赖图 + 错误传播模型（对应 provenance）
2. **机制**：[你的指标] + 分级验收 + 定位 + 最小重算决策（对应 MES + MESReduce）
3. **系统 + 证据**：在 Real2Sim 管线上的原型、故障注入、真实/合成对照、成本—质量曲线

**实验设置**（照 Query-Guided 的三场景 + 四 baseline）：
- 数据：3–5 对象起步 → 8–12 对象；自然故障 + 受控注入故障（尺度漏转、USD 错引、掩码过度、坐标翻转）
- 场景：RLBL（真实踩坑记录）/ WCS（全错初始化）/ AVG（随机注入）
- baseline：B0 现有流程 / B1 固定全检查+缓存 / B2 通用修复搜索 / Q 本方法
- 指标：**交付可靠性 vs 检查/修复成本曲线**、检出率、误拒率、交付率、人工分钟数、runtime
- 消融：去掉依赖图/去掉最坏指标/去掉定位——对应它的 different baselines

---

## 3. 一个必须解决的"翻译"问题

Query-Guided 那篇之所以严谨，是因为它有**严格的数学定义**（provenance 公式、MES、risky tuples、最坏情况保证）。你不能直接说"我们借用了 provenance"——你需要给出你的版本：

| 它需要 | 你必须定义清楚（【本项目】待完成） |
|---|---|
| provenance 表达式 | 资产依赖图的形式化：节点=产物，边=生成关系，标签=检查状态与置信度 |
| 错误传播规则 | 什么条件下上游错误一定导致下游不可用（如尺度错→USD 尺寸错），什么条件下不传播 |
| 最坏情况指标 | 给定当前检查证据，资产最坏可能违反哪些任务要求——不是概率保证，是 worst-case（避免"没有证明的概率"） |
| 决策算法 | 在预算内选下一组检查/修复动作——哪怕先做启发式，也要有明确的目标函数和 baseline |

**这是你接下来最该花时间的地方**——把直觉变成可写进论文的形式化，而不是"我们做了个系统"。

---

## 4. 和已有工作的划界（把这三篇变成你的 related work，不是你的对手）

- **Query-Guided / CBAClean / SOUND**：数据验证与清洗在表格/序列数据上已有——**它们不是三维资产**，且没有跨表示依赖图；你在新数据域上做形式化和系统
- **TAMIS / ManiTwin**：异常检测 / 资产生成验证——**它们不做跨格式一致性、不追踪错误传播、不算局部重算成本**
- **DRE**：单 GLB 可用性分类——**它只看单文件**，不看资产管线依赖

---

## 5. 证据边界

1. 三篇论文 PDF 与全文已下载到 `refs_icde_quality/`，关键论断均来自原文；CBAClean/SOUND/CDMap/GaV 只有摘要或官方页，不能当原文引用。
2. 本框架是"怎么抄结构"的建议，不是保证中稿；ICDE 是否接受三维资产这个数据域仍是未知数。
3. 你的数据/环境前提（本机无数据、单对象、无真实工业使用证据）仍未解决，抄结构不能替代证据。
4. 形式化部分是待完成工作，不要写成已有成果。
