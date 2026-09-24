# Real2Sim-Grasping

> 用 3DGS 重建做成高保真仿真资产，在 Isaac Sim 里跑抓取，并量化 sim-to-real 差距。
> 版本：v1（B 方案 · 真值已知排练版） · 创建 2026-09-14

---

## 目录结构

```
Real2Sim-Grasping/
├── README.md          ← 本手册（方法论 + 步骤 + 排错）
├── scripts/           ← 可独立运行的脚本（粘进 Script Editor 执行）
│   ├── 01_list_ycb_assets.py      T1.1  列出 YCB 资产
│   ├── 02_diagnose_bbox.py        T1.1b 诊断物体尺寸/位置
│   ├── 03_setup_scene.py          T1.2a 建桌面 + 加载物体
│   ├── 04_render_orbit.py         T1.2b-1 移动相机 + 逐帧 capture（不碰光照）
│   ├── 05_export_poses.py         T1.2b-2 收集图像 + 导出位姿
│   ├── 06_setup_lighting.py       可选：脚本化布光（默认在编辑器里手动调）
│   └── 07_render_standalone.py    ★推荐：standalone 一键（建场景+渲染+位姿）
└── data/              ← 渲染数据与重建产物（生成中）
```

脚本与手册分离：手册讲**为什么**和**怎么做**，脚本是**可直接执行的代码**。

---

## 一、项目定位

**一句话**：用公开/自带的物体资产，通过 **3DGS 重建**做成高保真仿真资产，在 Isaac Sim 里跑**抓取任务**，并**量化"仿真渲染 vs 源渲染"的差距**。

**为什么先做 B 方案（YCB 自带资产）**：
源数据是干净的仿真模型，**相机位姿是 ground truth**（不用 COLMAP 估计），几何已知——所以能精确量化每一步的质量。先在"真值已知"的环境验证管线，再换真实数据（A 方案）只需替换第①步。

**最终目标**：产出一条**可复用的 real→sim 资产管线**，任何场景数据都能喂进去。

---

## 二、理论基础

| 论文 | 年份 | 作用 |
|---|---|---|
| **SplatSim** — *Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting* | 2024 | 方法论核心：用 3DGS 做仿真渲染器，替代传统光栅化，缩小视觉 gap |
| **GraspNet-1Billion** — *A Large-Scale Benchmark for General Object Grasping* | CVPR 2020 | 抓取位姿标注标准（A 方案数据源） |
| **3D Gaussian Splatting** — Kerbl et al. | SIGGRAPH 2023 | 重建方法基础 |

补充：**ACRONYM**（ICRA 2021，ShapeNet 抓取数据集）、**Domain Randomization**（Tobin et al., 2017）。

---

## 三、数据流动

```
┌─ ① 源数据（零下载）
│   /Isaac/Props/YCB/Axis_Aligned/*.usd（真实物体的 3D 扫描模型，带纹理）
│
├─ ② 多视角渲染（替代手机拍照）
│   环形相机 36 机位 → RGB + 精确位姿（ground truth，无需 COLMAP）
│   输出：images/ + transforms.json
│
├─ ③ 3DGS 重建
│   ns-train splatfacto → splat.ply
│
├─ ④ 资产化：回 Isaac Sim
│   splat.ply → mesh → USD + 凸包碰撞体 + 质量 + 摩擦 → 摆 Franka
│
├─ ⑤ 抓取执行
│   定位 → 抓取位姿 → Franka 执行 → 记录成功/失败
│
└─ ⑥ 评估
    ②的源渲染 vs ③的 3DGS 新视角渲染 → PSNR / SSIM / LPIPS + 抓取成功率
```

---

## 四、环境

| 项目 | 值 |
|---|---|
| Isaac Sim | 6.0.1（`C:\isaacsim`） |
| 系统 | Windows 11 / RTX 5060 8GB |
| 3DGS | nerfstudio splatfacto（truck 场景已跑通 30k step） |
| Python | `C:\isaacsim\python.bat`（standalone）或 Script Editor |

### 已验证的关键 API

```python
# 资产根
from isaacsim.storage.native import get_assets_root_path
root = get_assets_root_path()

# 加载物体（参数是 path，不是 prim_path）
from isaacsim.core.experimental.utils.stage import add_reference_to_stage
add_reference_to_stage(usd_path=..., path="/World/Franka")

# 控制 Franka（9 DOF = 7 手臂 + 2 夹爪，弧度）
from isaacsim.core.experimental.prims import Articulation
art = Articulation("/World/Franka")
art.set_dof_position_targets(np.array([0, -0.5, 0, -2.0, 0, 1.5, 0.7, 0.04, 0.04]))

# 物理初始化（必须）
from isaacsim.core.simulation_manager import SimulationManager
SimulationManager.initialize_physics()
```

---

## 五、实施计划

### T1：多视角渲染（①→②） · 约 1h

| 脚本 | 作用 | 验收 |
|---|---|---|
| `scripts/01_list_ycb_assets.py` | 列出 YCB 物体，确认路径 | 看到 21 个物体列表 |
| `scripts/02_diagnose_bbox.py` | 诊断物体尺寸/位置 | 打印 size (m) |
| `scripts/03_setup_scene.py` | 建桌面 + 加载饼干盒 | 视口看到物体 |
| `scripts/06_setup_lighting.py` | **可选**：一键布光（Dome 1000 / Sun 0），也可在编辑器手动调 | 打印两行强度 |
| `scripts/04_render_orbit.py` | 只移动相机 + 逐帧 capture（**不碰光照**） | `rendering started (async)` → `all frames done` |
| `scripts/05_export_poses.py` | 等 30-60s 后运行，收集图像 + 写位姿 | `rendered: 36`、`transforms.json: 36 frames` |

### ★ 推荐：standalone 一键流程（替代 03 + 04 + 05）

```
C:\isaacsim\python.bat C:\Users\theozhu\Desktop\拓展\Real2Sim-Grasping\scripts\07_render_standalone.py
```

一个进程跑完：建桌面 → 加载物体 → 布光 → 36 机位渲染 → 写 `transforms.json`。

**为什么更可靠**：Script Editor 跑在常驻 Kit 进程里，Replicator 的图节点、writer 绑定、协程在脚本结束后依然存活并污染后续运行（幽灵相机复活、帧号漂移、新旧图混合）。standalone 进程跑完即退出，②③ 层状态归零，且允许同步 `step()`（Kit 内禁用）。

**✅ T1.1 实测结果**：

- `assets root` = `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0`（云端按需加载）
- 21 个物体可用：`002_master_chef_can`、`003_cracker_box`、`004_sugar_box`、`005_tomato_soup_can`、`006_mustard_bottle`、`007_tuna_fish_can`、`008_pudding_box`、`009_gelatin_box`、`010_potted_meat_can`、`011_banana`、`019_pitcher_base`、`021_bleach_cleanser`、`024_bowl`、`025_mug`、`035_power_drill`、`036_wood_block`、`037_scissors`、`040_large_marker`、`051_large_clamp`、`052_extra_large_clamp`、`061_foam_brick`
- **发现**：`/Isaac/Props/YCB/Axis_Aligned_Physics/` 是带物理属性的版本 → T3 可参考其参数
- **选用**：`003_cracker_box`

**✅ T1.1b 实测（003_cracker_box）**：
```
size (m): [0.164, 0.213, 0.072]      # 16.4 × 21.3 × 7.2 cm
min (m):  [-0.082, -0.107, -0.036]   # 中心在原点 → 一半嵌桌面，需抬高 0.036
```

**坐标系转换**（渲染脚本已内置）：USD 是 Z-up / 相机看 −Z（OpenGL），nerfstudio 是 Y-down / 相机看 +Z（OpenCV）。转换：`FLIP = diag(1, -1, -1, 1)`。

---

### T2：3DGS 重建（②→③） · 约 1h

```bash
ns-train splatfacto --data <数据目录> --output-dir <输出> --max-num-iterations 30000
ns-export gaussian-splat --load-config <config.yml> --output-dir <导出目录>
```

**验收**：`splat.ply` 生成；viewer 看新视角，物体清晰无浮空噪点。

> 环境：Windows + Blackwell sm_120 需 `TORCH_CUDA_ARCH_LIST=12.0`、`VSLANG=1033`（已固化为 bat）。

---

### T3：资产化 + 物理（③→④） · 约 1.5h

```python
UsdPhysics.CollisionAPI.Apply(obj)
UsdPhysics.MeshCollisionAPI.Apply(obj)
UsdPhysics.MeshCollisionAPI(obj).GetApproximationAttr().Set("convexHull")   # 凸包，比三角网格快
UsdPhysics.RigidBodyAPI.Apply(obj)
UsdPhysics.MassAPI.Apply(obj).GetMassAttr().Set(0.5)
```

**mesh 提取三条路**：TSDF fusion（最省事）→ 泊松重建 → SuGaR/2DGS（质量最好）。

**降难度备选**：3DGS 不转 mesh，只渲染成背景图贴进场景，前景用简单几何体。

**验收**：物体在桌面不下沉不抖；抓取时物理正常。

---

### T4：抓取 + 评估（⑤→⑥） · 约 1.5h

| 指标 | 对比 | 工具 |
|---|---|---|
| PSNR / SSIM / LPIPS | T1 源渲染 vs T2 的 3DGS 渲染 | `skimage` / `lpips` |
| 抓取成功率 | 仿真执行 N 次 | 自行统计 |

---

## 六、A 方案扩展（后续）

替换第①步为真实数据即可，T2-T4 完全复用（位姿改用 COLMAP 估计）：

| 数据集 | 用途 |
|---|---|
| **GraspNet-1Billion** | 真实桌面 RGB-D + 10 亿抓取位姿（抓取最权威） |
| **ScanNet / Replica / HM3D** | 室内场景扫描（导航/重建） |
| **YCB-Video** | YCB 物体视频 + 6DoF 位姿（轻量） |

---

## 七、排错指南

| 现象 | 原因 | 解法 |
|---|---|---|
| `add_reference_to_stage() got unexpected keyword 'prim_path'` | 参数名错 | 用 `path=` |
| S3 URL 加载失败 | 手写路径版本不对 | 用 `get_assets_root_path()` 拼 |
| `ApplyMultipleApplyAPI` 报错 | Articulation 不在 `/World/` 下 | 物体放 `/World/Franka` |
| `PhysxJointDriveAPI` 不存在 | Isaac 6.0 无此 API | 用 `Articulation.set_dof_position_targets` |
| 机械臂持续摆动 | PhysX 持有旧目标 | `set_dof_position_targets(np.zeros(9))` |
| 改 USD 属性后仿真无变化 | DriveAPI 是 metadata | 走 Articulation API 或 Stop→改→Play |
| 物体看不见 / 半埋桌面 | YCB 原点在几何中心 | 抬高 `z = size_z/2`，先跑 `02_diagnose_bbox.py` |
| 3DGS 重建后物体倒/歪 | 坐标系没翻转 | 检查 c2w 的 FLIP 矩阵 |
| `rendered: 0`（sleep 后立刻查） | `time.sleep` 阻塞主循环，而 Replicator 在主循环里渲染 | 拆两步：04 启动 → 等 30-60s → 05 收集 |
| `rendered: 0`（等很久仍 0） | S3 资产首次加载慢 / orchestrator 未启动 | 等更久或重跑 04 |
| `'BasicWriter' object is not callable` | Isaac 6.0 的 `WriterRegistry.get` 返回实例 | 实例上调 `.initialize(output_dir=..., rgb=True)` |
| 视口正常但渲染全白 | 视口有自动曝光，writer 无 tone mapping，HDR>1 直接 clamp 成白 | 调相机 **exposure:fStop**（进光 ∝ 1/fStop²，f/1→f/32 = 亮度 ÷1024）；或调低灯光强度。曝光归 GUI，脚本不碰 |
| 渲染里几何全部消失 | 相机 `clippingRange` 默认近裁 1m，物体在 0.6m 被剔除 | `GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))` |
| 相机朝向不对 | USD `Matrix4d` 行主序 + 平移在最后一行，列向量矩阵需转置 | `Gf.Matrix4d(*m.T.flatten())` |
| 改了脚本但行为没变 | Script Editor 内容**不会自动同步文件** | 每次改完重新打开文件、全选复制、粘进 Script Editor |

---

## 八、进度

| 任务 | 状态 | 完成时间 |
|---|---|---|
| T1.1 确认 YCB 路径 | ✅ 完成 | 2026-09-14 |
| T1.1b 诊断物体尺寸 | ✅ 完成 | 2026-09-14 |
| T1.2a 摆场景 | ⬜ 待做 | |
| T1.2b 36 视角渲染 | ⬜ 待做 | |
| T2 3DGS 重建 | ⬜ 待做 | |
| T3 资产化 + 物理 | ⬜ 待做 | |
| T4 抓取 + 评估 | ⬜ 待做 | |
