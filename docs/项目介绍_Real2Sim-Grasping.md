# Real2Sim-Grasping · 从 RGB 照片到机器人抓取的闭环

> 一句话：**只用多视角 RGB 照片**，用 3DGS 重建出物体网格，做成 Isaac Sim 里能被 Franka 抓起来的仿真资产，
> 并量化它与真值资产差多少。
>
> 为什么能量化：渲染端是 Isaac Sim，位姿和深度都是真值，所以可以拿"真值网格"当尺子去量"重建网格"。

---

## 一、做了什么

1. **多视角渲染（T1）**：Isaac Sim 里 99 个机位环绕拍 YCB 饼干盒，导出 RGB + 真值位姿 + 真值深度。
2. **3DGS 训练（T2）**：nerfstudio `splatfacto` 训 30k 步 → `splat.ply`。
3. **真值基准（T3）**：用**真值深度**做 TSDF，得到 GT 网格，当作评估用的"尺子"。
4. **闭环（T5，核心）**：让训好的 3DGS **自己渲染** 99 个视角的深度（纯推断，不碰真值）→ TSDF → Marching Cubes → 3DGS 网格，与 GT 网格算 Chamfer。
5. **抓取（T4）**：Franka + 物理回调状态机（open→hover→descend→close→lift→judge），物体被抬起 > 5 cm 判成功。

---

## 二、遇到的困难与怎么解决的

| 困难 | 解决 |
|---|---|
| gsplat 在 Windows 无预编译 wheel，JIT 编译失败且真错误被中文编码吞掉 | 单独预编译并缓存（`fix_gsplat.bat`）；运行时用**同一套环境变量**（`run_nerfstudio.bat`）保证缓存命中 |
| `ns-export` 全线崩溃（`get_outputs` 收到非 `Cameras` 参数时返回 `{}`） | 不用导出器，自己构造 `Cameras` 直接调 `model.get_outputs()` 取 depth |
| 训练开了 `auto-scale-poses`，深度是归一化单位不是米 | 从 `dataparser_transforms.json` 读 `scale` 还原，并与真值深度交叉验证 |
| 坐标系：transforms.json 是 OpenGL，Open3D TSDF 要 OpenCV | 加 `FLIP = diag(1,-1,-1)`，否则几何被镜像到相机后方（裁剪后 0 顶点） |
| 训练**没用 mask**，背景高斯混进模型 → 融进 TSDF 变成百万顶点团 | 用**世界坐标几何掩码**替代色度掩码：footprint + 桌面上方 3 mm |
| 色度掩码（`chroma>40`）误杀物体白色区域 → 网格大片破洞 | 判据从"够不够彩"换成"属不属于物体"（几何量，与颜色无关） |
| 掠射角下期望深度把壁面和桌面混合 → 底部"斗形"变形（误差 22 mm） | 对每帧掩码做 8 px **腐蚀**，剔掉轮廓污染带；Chamfer 7.87 → 3.00 mm |
| 想用数据众数估桌面高度 → z 分布多峰，估错 14 mm，切掉物体底部 | 改用资产先验（物体底面 = 桌面），失败估计保留作诊断输出 |
| `acc`（累积不透明度）实测**恒为 1**，分不开前景 | 放弃用它分前景（背景桌面同样不透明），只当有效性检查 |
| 单帧掩码只看得到物体一部分，担心结果残缺 | 实测 63 帧并集：每点平均被 23 帧看到，顶面 36 帧，从未观测的点仅 0.45% |
| 抓取用协程会挂在第一个 `await`（Kit 的 async_engine 驱动不了） | 改成订阅 physics step 的状态机，成功/失败/超时/**崩溃**都自动 unsubscribe |
| 顶点色发白 | 换用 GS 自渲染色（+9%）；剩余判定为 Isaac 显示端，非数据问题 |

---

## 三、现在的数字

| 指标 | v1（色度掩码） | v2（几何掩码） | 真值 |
|---|---|---|---|
| **Chamfer** | 5.00 mm | **3.00 mm** | — |
| 边界边（破洞） | 4566 | **1136（−75%）** | 378 |
| bbox x×y×z (m) | 0.174×0.221×0.042 | **0.164×0.213×0.036** | 0.164×0.213×0.042 |

产物：`3DGS/exports/ycb/gs_mesh_v2_mask-footprint+worldz_color-gsrgb_voxel2mm.obj`

---

## 四、还没做完

- **抓取脚本 17 挂的仍是 GT 网格**，没换成 3DGS 重建的 v2 → GT vs 3DGS 的**抓取成功率对比**还没做（TODO P0）
- 底部 6 mm 缺失（物体—桌面接触区）、顶面小凹坑（GS 深度空洞，RGB-only 上限）
- 发白若在 MeshLab 里也淡，再查 Isaac 材质 / xatlas 烘贴图

---

## 五、文档索引

| 文档 | 讲什么 |
|---|---|
| `README.md` | 方法论 + 步骤 + 排错指南 |
| `docs/TODO.md` | 待办与技术结论（勿重复踩坑） |
| `docs/Lesson06_TSDF融合与网格提取.html` | TSDF / Marching Cubes 原理 |
| **`docs/Lesson07_从色度掩码到几何掩码.html`** | **本次重建的两次缺陷定位与修复全过程** |
| `ECOSPLATNOTES/` | EcoSplat（前馈式 3DGS）源码学习笔记 Ch.01–05 |
