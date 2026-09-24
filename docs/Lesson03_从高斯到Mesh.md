# Lesson 03 · 从高斯到 Mesh：导出器崩溃、泊松膨胀、TSDF 翻盘

> 阶段：T3（3DGS → mesh）
> 环境：nerfstudio 1.1.5 / gsplat 1.4.0 / open3d 0.19.0 / Windows 11
> 结果：拿到 **0.2 mm 精度**的真实尺度 mesh（0.164 × 0.213 m，与 YCB 官方尺寸一致）

---

## 1. 一句话总结

`ns-export` 撞上 nerfstudio 1.1.5 的 bug 全线崩溃 → 绕过导出器直读 checkpoint 取高斯参数 →
泊松重建把盒子**膨胀圆化**了 10 mm → 改用 **TSDF 融合真值深度图**，精度回到 0.2 mm。

---

## 2. 问题时间线

| # | 现象 | 真因 | 解法 |
|---|---|---|---|
| 1 | `ns-export tsdf` 报 `Could not find rgb in the model outputs` | 导出器 bug：`get_outputs()` 收到非 `Cameras` 直接返回 `{}` | 不用导出器，直读 checkpoint |
| 2 | 泊松重建后盒子 0.234×0.234（真值 0.164×0.213） | 泊松对锐利边角物体会**外扩 + 圆化** | 改用 TSDF |
| 3 | 尺度标定误差 102 mm | floaters 污染 `z_max`，标定基准点错了 | TSDF 路线 + 空间裁剪，不需要标定 |
| 4 | 裁剪后 mesh 碎成小块（最大块 1442 面） | `select_by_index` 对 mesh 是**顶点**索引，我传了三角形索引 | 改用 `remove_triangles_by_mask` |

---

## 3. 问题 1：导出器全线崩溃

### 现象

```
Called get_outputs with not a camera
╔═══ Error ═══╗
  Could not find rgb in the model outputs
  Please set --rgb_output_name to one of: dict_keys([])
```

**两种模式都失败**：`tsdf` 和 `gaussian-splat` 报完全相同的错。

### 定位

搜源码找到唯一匹配（`splatfacto.py:539`）：

```python
if not isinstance(camera, Cameras):
    print("Called get_outputs with not a camera")
    return {}                       # ← 空字典，下游自然找不到 rgb
```

**结论：这是 nerfstudio 1.1.5 导出器的 bug，和我们的数据、位姿、内参全都无关。**
（同一个环境早期导出 truck 成功过，是因为那时的调用路径不同。）

**教训**：报错提示 "Please set --rgb_output_name to one of: dict_keys([])" 是在骗人——可选列表是**空的**，设什么都没用。看到空字典就该去搜源码，而不是试参数。

---

## 4. 绕行方案：直读 checkpoint

高斯参数本来就是 checkpoint 里的张量，何必走导出器：

```python
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["pipeline"]
means   = sd["_model.gauss_params.means"]        # (N, 3)
scales  = torch.exp(sd["_model.gauss_params.scales"])   # exp() 还原世界单位
opacity = torch.sigmoid(sd["_model.gauss_params.opacities"])
fdc     = sd["_model.gauss_params.features_dc"]  # 球谐 DC 项 (N,1,3)
```

两个要点：

1. **key 前缀因版本而异**（`_model.` vs `model.`）→ 用 `endswith("gauss_params.means")` 自动探测
2. **SH DC → RGB**：`rgb = clamp(0.5 + 0.28209479177387814 * fdc, 0, 1)`

### 意外收获：更干净的 floater 过滤

直读参数后可以按**物理属性**过滤，比几何滤波准得多：

```python
keep = (opacity > 0.5) & (max_scale < quantile(max_scale, 0.90))
```

结果：**131.9 万高斯 → 68.8 万**（剔掉 48%，全是低不透明度 / 超大尺寸的 floaters）

对应脚本：`scripts/13_export_gaussians.py`

---

## 5. 问题 2：泊松重建把盒子"吹胖了"

用 GT 真值点云做泊松（输入 bbox 完全正确：x∈[-0.082,0.082], y∈[-0.106,0.106]）：

```
input  bbox: 0.164 × 0.213 × 0.042   ← 完全正确
poisson bbox: 0.234 × 0.234 × 0.074  ← 每边肿 ~10 mm，还变成正方形
x/y error: 70.3 mm -> CHECK
```

**根因**：泊松求解的是平滑隐式场，天生会**外扩并在边角处圆化**。对雕塑、人脸这类有机形状是优点，对**平板纸盒**是灾难——抓取需要精确尺寸，10 mm 误差足以让夹爪抓空。

**结论：泊松不适合工业零件/盒状物体。**

---

## 6. 问题 3：尺度标定被 floaters 污染

3DGS 的 mesh 处在 nerfstudio 的**归一化空间**（训练时 `auto-scale-poses` 缩放过），尺寸无意义。
我的标定思路是：用已知几何反解

```
scale = 0.047 / (z_max - z_table)
```

其中 `z_table` 取 z 直方图众数（桌面），`z_max` 取最高点（盒顶）。

**失败原因**：`z_max` 取到的是某个**floater 的最高点**，不是盒顶 → scale 算成 0.0241 → x/y 误差 102 mm。
而且最大连通分量有 **337 万面（几乎全部）**，说明泊松把噪声和物体连成了一整个壳，连通分量清理完全失效。

---

## 7. 终解：TSDF 融合真值深度

### 为什么 TSDF 更好

| | 泊松 | TSDF |
|---|---|---|
| 原理 | 从点云 + 法线解隐式场 | 多视角深度**投票融合**成体素场 |
| 边界 | 外扩、圆化 | **贴合，保持平面和锐边** |
| 对噪声 | 敏感（噪声被包进曲面） | 鲁棒（多视角投票天然抑制） |
| 适合 | 有机形状、雕塑 | **工业件、盒状物体** |

### 输入（全部是真实物理量，无需标定）

| 输入 | 来源 |
|---|---|
| 深度图 | Isaac Sim `distance_to_image_plane`，单位米 |
| 位姿 | `transforms.json`（仿真真值，OpenCV 约定） |
| 内参 | `transforms.json`（真实焦距 342.07） |

坐标约定天然一致：**open3d TSDF 用 OpenCV 约定（y 下、z 前），我们的 c2w 正是这个约定**，
所以外参直接 `inv(c2w)`，不需要任何转换。

```python
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.002, sdf_trunc=0.012,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
for fr in frames:
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth_img, depth_trunc=0.8, depth_scale=1.0,   # 深度已是米
        convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, np.linalg.inv(c2w))
mesh = vol.extract_triangle_mesh()
```

参数：体素 2 mm、`sdf_trunc` 12 mm、`depth_trunc` 0.8 m（剔除远处背景）

前置改动：`07_build_init.py` 增加 `COPY_DEPTH`，把真值深度图复制到 `data/ycb/depth/`，与图像一一对应。

---

## 8. 问题 4：open3d 裁剪 API 的坑

```python
# ❌ 错：对 TriangleMesh，select_by_index 按【顶点】索引
mesh.select_by_index(np.where(triangle_keep)[0])
#  → [SelectByIndex] indices contains index 1169331 out of range
#  → 只留下零星碎块，最大连通分量 1442 面

# ✅ 对：按三角形掩码删除
mesh.remove_triangles_by_mask(~triangle_keep)
mesh.remove_unreferenced_vertices()
```

**教训**：open3d 里 `PointCloud` 和 `TriangleMesh` 的同名方法语义可能不同。看到
`index out of range` 警告就要立刻怀疑索引类型。

---

## 9. 最终结果

```
raw mesh:        671928 verts / 1337714 tris
after crop:       16646 verts /   32912 tris
largest component: 32912 tris（单一连通体，无碎片）

GT mesh bbox:  x=0.164  y=0.213  z=0.042
YCB official:  x=0.164  y=0.213  z=0.072
x/y error:     0.2 mm  -> OK
```

z=0.042 而非 0.072，是因为**盒子底部陷在桌面下 25 mm**（Lesson 01 记录过），可见高度本就只有 ~0.047 m，正确。

脚本：`scripts/15_tsdf_from_depth.py`
产物：`3DGS/exports/ycb/gt_mesh_tsdf.obj`

---

## 10. 经验总结

1. **框架的"使用建议"可能是错的**：`Please set --rgb_output_name to one of: dict_keys([])` —— 选项是空的，问题根本不在参数。空列表/空字典是强烈的"去读源码"信号。
2. **遇到问题先看能不能绕过中间层**：导出器坏了，但高斯参数就躺在 checkpoint 里，`torch.load` 直读反而拿到了**更多控制权**（能按 opacity/scale 过滤）。
3. **重建算法要按物体类型选**：泊松为有机形状设计，工业件/盒状物体用 TSDF。选错算法，输入再准也没用（这次输入 bbox 完全正确，输出仍肿了 10 mm）。
4. **标定基准点必须抗污染**：用 `max()` / `min()` 这类极端值做基准，等于把命运交给噪声。优先用**已知几何 + 空间裁剪**，而不是从含噪数据里挑极值。
5. **同名 API 在不同类型上语义可能不同**（open3d `select_by_index`）。

---

## 11. 遗留待办

3DGS 重建出的 mesh 仍受**归一化尺度**困扰。根治办法是训练时加：

```
nerfstudio-data --auto-scale-poses False
```

这样导出的 mesh 直接就是真实米制，无需标定。注意同步评估
`--pipeline.model.densify-grad-thresh`（场景尺度变小后梯度量级会变，可能需要调小）。

届时即可做完整量化闭环：**3DGS mesh vs GT mesh 的 Chamfer distance + 抓取成功率对比**。
