# 论文资料与 GS → Mesh → 仿真基线候选

核查日期：2026-11-16。

本目录汇总两类资料：

1. **ICDE 工业赛道往届录用论文**：用于判断论文贡献与实验证据的要求，已下载三篇 PDF。
2. **三维重建与仿真的对比方法**：用于选择项目的实验 baseline，重点补充 2025–2026 年候选。

**阅读边界**：下面的指标来自作者论文，不是本项目复现结果。代码入口已核对，但未在本机安装或运行这些候选。新近论文不自动等于所有任务上的 SOTA；不同数据集和评测协议的成绩不能直接排名。

## 一、已经下载的 ICDE 工业赛道论文

| 论文 | 本地 PDF | 提取正文 | 重点阅读 |
|---|---|---|---|
| Deep Learning with Spatiotemporal Data: A Deep Dive into GeotorchAI，ICDE 2024 | [GeoTorchAI](ICDE2024_GeoTorchAI.pdf) | [带 PDF 页码的正文](ICDE2024_GeoTorchAI.txt) | 第 9–12 页：数据预处理、训练与端到端评估 |
| Xorbits: Automating Operator Tiling for Distributed Data Science，ICDE 2024 | [Xorbits](ICDE2024_Xorbits_arxiv_v1.pdf) | [带 PDF 页码的正文](ICDE2024_Xorbits_arxiv_v1.txt) | 第 10–12 页：规模、baseline、动态分块与计算图消融 |
| MLKV: Efficiently Scaling up Large Embedding Model Training with Disk-based Key-Value Storage，ICDE 2025 | [MLKV](ICDE2025_MLKV_arxiv_v1.pdf) | [带 PDF 页码的正文](ICDE2025_MLKV_arxiv_v1.txt) | Section IV：训练任务、存储开销、生产数据与消融 |

### 从这些论文学什么

- **GeoTorchAI**：已有算法的组合也可以形成系统贡献，但要解决共性数据处理瓶颈。论文对比了 140 万至 2.5 亿条记录的预处理时间和内存；明确在单机上比较，不应误称为集群扩展实验。模型包含重复运行与波动报告。Table VIII 的一个设置从 32,977 秒降到预处理 1,276 秒＋训练 22,429 秒＝23,705 秒，收益计算包含预处理成本。
- **Xorbits**：解释动态分块为何缓解数据倾斜和内存溢出。测试包括 20–358 GB 工作负载、最多 16 个 worker 实例；有 baseline、消融和 API 兼容性评估。作者另报告生产集群达到 5,000 CPU 核，这与实验集群规模不同。
- **MLKV**：创新落在机器学习的数据访问与存储，而不只是更换模型。评估包括 7 个数据集、3 类任务、吞吐与模型质量，以及陈旧度控制和预取消融；其中两个数据集来自 eBay 生产业务。

**这些规模、倍数和数据集数量不是 ICDE 官方最低要求，也不是录用保证。**

### 来源与版本

- 官方录用名单：[ICDE 2024](https://icde2024.github.io/industry.html)、[ICDE 2025](https://ieee-icde.org/2025/industry-and-application-papers/)。
- GeoTorchAI：作者公开的 [PDF](https://kanchanchy.github.io/files/GeoTorchAI-ICDE-2024.pdf)，本地副本共 14 页。
- Xorbits：[arXiv v1](https://arxiv.org/abs/2401.00865v1)，本地副本共 13 页。
- MLKV：[arXiv v1](https://arxiv.org/abs/2504.01506v1)，本地副本共 8 页。
- 后两份为固定版本的作者预印本，可能与最终出版版本不同。全部来源、版本与阅读证据见 [sources.json](sources.json)。
- CDMap 仅核对了官方录用名单和[机构摘要页](https://pure.ecnu.edu.cn/en/publications/cdmap-complementarity-and-disparity-aware-map-inference-quality-e)，未下载全文，不作为全文实验依据。

论文及代码的权利和许可归各自作者、出版方及原项目所有；本目录的资料整理不改变其许可。公开可读不等于可以任意再许可。

## 二、2025–2026 年重点基线候选

### 1. TopoSurfel，2026

**论文**：[TopoSurfel: Closing the Loop between Gaussian Surfels and Meshes for Surface Reconstruction](https://arxiv.org/abs/2608.20687)

**代码**：[Fan-Treasure/TopoSurfel](https://github.com/Fan-Treasure/TopoSurfel)

- 作者仓库标注 SIGGRAPH Asia 2026 / TOG；该会议信息来自作者侧声明，本次未独立核验最终出版记录。
- 核心是训练中的闭环：surfel 生成可微代理 mesh，mesh 的深度、法线及表面结构反向约束 surfel，并引导增删。最终输出仍可通过 TSDF 融合得到。
- 论文同表报告：DTU 平均 CD 为 **0.51 mm**，PGSR 为 **0.53 mm**；Tanks & Temples 平均 F1 为 **0.52**，PGSR 为 **0.50**。
- 这不是所有方法都被超过：同一 DTU 表中，使用额外单目深度先验的 GeoSVR 为 0.47 mm。训练耗时也不是最快。
- 适合本项目的密集多视图、表面几何和 mesh 质量对比。
- **接入注意**：官方工作流需要初始三角网格 `mesh_init.ply`；不能偷偷以 GT mesh 初始化。应记录初始网格的生成方式，并把这部分成本计入。
- **复现成本**：README 列出 Python 3.10、PyTorch 2.5.1、CUDA 12.1，以及 nvdiffrast、PyTorch3D、Kaolin 和需要编译的子模块；本机兼容性尚未验证。

### 2. CoMe，2026

**论文**：[Confidence-Based Mesh Extraction from 3D Gaussians](https://arxiv.org/abs/2603.24725)

**代码**：[r4dl/CoMe](https://github.com/r4dl/CoMe)

**项目页**：[CoMe](https://r4dl.github.io/CoMe/)

- 作者仓库和项目页标注 ECCV 2026；本次未独立核验会议录用清单。
- 通过自监督置信度、颜色与法线方差约束、外观处理和增密调整，缓解高斯用错误几何解释视角相关外观的问题。
- 不依赖额外大型预训练单目深度／法线模型；多视图输入不等于额外的跨视图一致性损失。
- 论文 Tanks & Temples 同表平均 F1：**GOF 0.453、PGSR 0.496、CoMe 0.521**。
- 不是全面第一：论文 DTU 同表平均 CD 为 CoMe 0.65、PGSR 0.55；该网页表头未明确标注长度单位，不能把该数值直接与本项目毫米误差比较。
- 单对象提供 TSDF 提取；大场景提供四面体提取。
- **接入注意**：这是包含训练改进的完整方法，不是把任意 splatfacto checkpoint 塞进去就能得到论文效果的后处理插件。要使用作者的网格训练配置；README 的最简训练命令并未开启全部相关项。
- **复现成本**：作者测试环境为 Ubuntu 22.04＋CUDA 12.1，包含 CUDA 光栅化／SSIM 扩展；四面体路线还有额外编译依赖。

### 3. Sparse2DGS，2025

**论文**：[Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views](https://arxiv.org/abs/2504.20378)

**代码**：[Wuuu3511/Sparse2DGS](https://github.com/Wuuu3511/Sparse2DGS)

- 作者仓库标注 CVPR 2025。
- 使用预训练 CLMVSNet 的深度和特征辅助初始化与几何优化，最终通过 TSDF 输出 mesh。
- 论文 DTU **3 视角**同表平均 CD：2DGS 2.81、PGSR 2.08、Sparse2DGS **1.13**。正文及表头未明确写长度单位，转述时保留论文口径。
- 是该表所比较的 GS 方法中的领先结果，而非所有路线中的最低 CD；例如同表 NeuSurf 为 0.99。
- **适用条件**：研究少视角、少采集成本时重点加入。不能把 3 视角成绩直接用于本项目 99 视角设置；正文未报告 Sparse2DGS 本身的 dense-view 指标。
- **接入注意**：需披露额外 MVS 预训练先验、权重与初始化成本；不能与无先验方法混为同一输入条件。

### 4. Geometry-Grounded Gaussian Splatting / Stochastic Solids，2026

**论文入口**：[arXiv:2601.17835](https://arxiv.org/abs/2601.17835)

**项目页**：[Geometry-Grounded Gaussian Splatting](https://baowenz.github.io/geometry_grounded_gaussian_splatting/)

**代码**：[HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids)

- 原仓库地址 `Geometry-Grounded-Gaussian-Splatting` 当前抓取结果显示为上述新仓库名。
- 仓库当前论文标题为 *Grounding Gaussian Splatting Geometry with Stochastic Solids*，标注 SIGGRAPH Asia 2026；项目页仍使用原标题并标注 arXiv 2026。应在正式引用时再核对最终版本。
- 作者将高斯与随机实体联系起来，以更一致的几何解释改善深度和网格提取。
- 已列出训练、DTU 网格提取及 TnT 四面体提取入口。
- **核实边界**：本次项目页与 README 没有可提取的定量结果表，因此作为近期候选，不直接宣称其最优。
- README 测试组合涉及 CUDA 12.8／13.0、Python 3.12 及多个编译扩展，环境接入成本可能较高。

## 三、经典基线仍应保留

| 方法 | 官方代码 | 作用及边界 |
|---|---|---|
| 原始 3DGS 深度＋TSDF | 本项目 `22`＋简化的 `23b` 路线 | 去掉本项目几何过滤，获得透明的基础对照 |
| 2DGS，SIGGRAPH 2024 | [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) | 表面元表示、几何约束与 TSDF；是经典参考，不代表 2026 全任务最优 |
| GS2Mesh，ECCV 2024 | [yanivw12/gs2mesh](https://github.com/yanivw12/gs2mesh) | GS 新视角立体图像＋预训练 DLNR 深度＋TSDF；需要披露额外学习先验 |
| SuGaR，CVPR 2024 | [Anttwo/SuGaR](https://github.com/Anttwo/SuGaR) | 表面对齐＋Poisson＋可选联合细化；提供纹理 mesh；官方流程以 vanilla 3DGS 为基础 |

本项目 v1 色度掩码与 v2 几何掩码可保留作内部对照／消融，但仅做 v1/v2 不能证明优于公开强方法。

## 四、进入机器人仿真的系统参考

**PolaRiS**：[项目页](https://polaris-evals.github.io/) · [代码](https://github.com/arhanjain/PolaRiS) · [环境 Hub](https://huggingface.co/datasets/owhan/PolaRiS-Hub)

- 官方 README 展示视频 → splat＋mesh（作者采用 2DGS）→ 组装 USD → Isaac Sim／Isaac Lab 策略评测的流程。
- 视觉仍保留 splatting，不能默认等价于本项目的彩色 mesh 渲染。
- README 没有充分解释每个可抓物体的视觉网格和碰撞几何来源；项目页提到的其他资产生成步骤需单独核实。
- 有研究代码与环境下载入口，但本次没有下载环境、检查资产或运行评测。
- 适合作重要相关工作和系统参考；不能直接拿它的任务成功率与本项目比较。

*Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data*：[论文](https://arxiv.org/abs/2506.04120)

- 属于与物理场景重建接近的 2025 工作。
- 本次 arXiv 摘要页未提供具体代码仓库，不能称已开源；暂列阅读候选，不列可运行 baseline。
- 注意不要与另一篇同名 SplatMesh 三维编辑论文混淆。

## 五、本项目建议的对照组与公平协议

优先级：**原始 3DGS＋TSDF → 2DGS → CoMe → TopoSurfel**；少视角预算实验再加入 Sparse2DGS。

1. 所有方法使用同一批 RGB、相机与对象划分；保留未参与调参的测试对象。
2. 披露相机真值、点云初始化、分割、尺寸和桌面高度先验。当前项目用过真值深度初始化，不能在其他方法只用 SfM 点云时直接声称公平比较。
3. 新方法按作者工作流训练，不默认支持 splatfacto checkpoint；冻结源码版本、配置、种子和数据划分。
4. 统计端到端成本，包含初始化、训练、深度推理、网格提取、后处理与人工操作；作者报告的训练时间不等于总耗时。
5. 输出网格统一到米制坐标，再使用相同碰撞生成规则、质量、摩擦、控制器和起始状态做配对抓取。质量／摩擦并非上述几何重建方法自动恢复的真实属性。
6. 同时报告几何准确度、完整度、任务相关接触区域误差、重复抓取成功率与置信区间、时间和峰值资源。
7. 原始 YCB 资产可作参考／理想资产对照；当前 `gt_mesh.obj` 是真值深度 TSDF 产物，本身也可能缺底面，不是毫无缺陷的绝对真值。
8. `17_setup_grasp_asset.py` 当前默认加载 GT 网格，实验必须记录实际使用资产，不能仅凭场景里一次成功证明 GS 网格的收益。

## 六、对论文方向的影响

- **仅“加置信度”**与 CoMe 已有内容重叠；**仅“让 mesh 反向约束高斯”**与 TopoSurfel 重叠，不能直接当作新贡献。
- 更值得继续验证的是：任务相关资产质量检查、预算约束下选择性补观测／处理，以及统一物理条件下的交互收益。
- “GS＋mesh＋Isaac 跑通”已有相关系统，论文需证明共性瓶颈、机制和收益，而不只展示演示。
- 真正是否有新颖性，需要进一步查完整相关工作和实际复现；本目录不提供录用或效果保证。

## 七、ICDE 2027 投稿条件备忘

[官方工业与应用赛道 CFP](https://icde2027.github.io/cf-industry-track.html)：至少一名作者具有真实的非学术机构 affiliation；此条件本身不足以使论文属于工业赛道。正文需阐明数据管理领域的工业创新。

- 投稿截止：**2026-12-01 23:59 Pacific Time**。
- 短文：6 页正文；长文：12 页正文；参考文献不限。
- 单盲审稿，保留作者与机构信息。
- 官网未规定最低数据规模、生产部署时长或用户数量。实验规模建议不等于官方门槛。

本项目当前已有工程原型与单对象结果，但多对象、重复抓取、工业使用和数据工程贡献仍待补强。较新的 baseline 能提高评估可信度，不能替代这些证据。
