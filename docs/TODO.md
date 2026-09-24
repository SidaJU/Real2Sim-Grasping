# 待办 / 已知未完成项

> 当前状态：T1 渲染 ✅ · T2 训练 ✅ · T3 GT 资产 ✅ · **T4 抓取进行中**
>
> 重要：**现在用的 mesh 是 GT（真值深度 TSDF 建的），不是 3DGS 重建的。**
> 任何对外表述都必须说清这一点，不能把它当作重建成果。

---

## P0 · 严格闭环未完成（最关键）

### 1. 真·3DGS mesh 尚未产出

**现状**：`exports/ycb/gt_mesh_tsdf.obj` 是拿 **Isaac Sim 真值深度**融出来的 Ground Truth，
**没有经过 3DGS**，精度 0.2 mm 代表"上界"，不是重建成绩。

**正确闭环应该是**：

```
① 用训练好的 3DGS 模型【自己渲染】99 个视角的深度图
      ↑ 关键：深度是模型推断的，不是仿真真值
② 渲染深度 + 位姿 → TSDF 融合 → 3DGS mesh
③ 3DGS mesh vs GT mesh 算 Chamfer distance → 量化重建误差（mm）
④ 两个 mesh 各跑 N 次抓取 → 对比成功率
```

**卡点**：`ns-export` 在 nerfstudio 1.1.5 上全线崩溃（`splatfacto.get_outputs()`
收到非 `Cameras` 参数时 `return {}`，见 `Lesson03`）。

**绕过方案**：不用导出器，直接调模型渲染：

```python
# 构造正确的 Cameras 对象（而不是 tuple/ndarray），即可拿到 depth
outputs = pipeline.model.get_outputs(camera)   # camera 必须是 Cameras 类型
depth = outputs["depth"]
```

参考：`nerfstudio/scripts/render.py` 里构造 Cameras 的方式。

**做完之后才能得到那句最值钱的话**：
> "用仿真数据重建的资产，抓取成功率达到真值资产的 X%，重建误差 Y mm"

---

## P1 · 训练配置

### 2. 3DGS 训练应关闭自动缩放

当前训练默认开启 `auto-scale-poses`，场景被归一化 → 导出 mesh 尺寸无意义，
导致 Lesson 03 里那次标定失败（误差 102 mm）。

重训时应加：

```
nerfstudio-data --auto-scale-poses False
```

注意：场景尺度变小后梯度量级会变，需同步评估
`--pipeline.model.densify-grad-thresh`（可能需要调小）。

---

## P2 · 场景与数据

### 3. 饼干盒陷进桌面 2.5 cm

盒子底面 z=0，桌面 z=0.0252 m → 下缘 1/3 埋在桌面里，永远拍不到。
（真实场景物体放桌上底部同样不可见，属物理真实；但这里是**穿模**。）

修法：把物体抬高 0.025 m，使底面与桌面齐平。

### 4. 曝光由自动曝光控制

采集时自动曝光跟随画面内容漂移（Lesson 03 记录）。
当前用**后处理增益归一化**缓解（漂移 8.49 → 2.66）。
更彻底的做法是在渲染端锁定曝光。

### 5. 物体本身偏简单

单个 YCB 盒子，纹理丰富但形状简单。若想让重建更有说服力，可考虑
增加物体（多物体场景）或换更难的形状（薄壁、反光件）。

---

## 已确认的技术结论（勿重复踩坑）

| 结论 | 依据 |
|---|---|
| 盒状物体用 **TSDF**，不用泊松 | 泊松把 0.164×0.213 肿成 0.234×0.234 |
| Kit 渲染**各向同性**，`fy = fx` | 桌面水平性判据：0.1 mm vs 15 mm |
| 中文路径会导致 Python 传参崩溃 | `UnicodeDecodeError` at `read_triangle_mesh`；一律用 `C:\isaacsim_data\` |
| 初始点云用**真值深度反投影** | 比 COLMAP 稠密、精确、秒级（Lesson 01） |
| gsplat 需预编译并缓存 | 否则训练时 JIT 编译失败（Lesson 02） |
