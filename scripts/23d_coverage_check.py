"""How many frames actually observe each surface point?

The per-frame mask (debug_v2_mask_f*.png) is only a PARTIAL view of the object:
one camera sees the top face plus one or two walls, and erosion eats another
8 px ring. The mesh is still complete because TSDF fuses 63 such frames.

This script measures that coverage directly:
  for every sampled surface point of the final mesh, count the frames in which
  the point projects into the kept mask AND the observed depth agrees with it
  (within DEPTH_TOL). That is "how many times was this point really seen".

Run:  C:\\Users\\theozhu\\miniconda3\\envs\\nerfstudio\\python.exe 23d_coverage_check.py
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_erosion

DATA = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb")
GS = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\exports\ycb\gs_depth")
NORM_JSON = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\outputs\ycb\splatfacto"
                 r"\2026-09-23_102733\dataparser_transforms.json")
MESH = Path(r"C:\isaacsim_data\mesh\gs_mesh_v2_for_compare.obj")   # ASCII copy

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
MASK_HALF_X, MASK_HALF_Y = 0.083, 0.107
TABLE_Z, MASK_Z_UP, TABLE_MARGIN = 0.030, 0.080, 0.003
ACC_MIN, DEPTH_TRUNC, MASK_ERODE = 0.10, 0.8, 8
SKIP_LOW_RING = True
DEPTH_TOL = 0.008          # 8 mm: observed depth must agree with the point
N_POINTS = 4000

SCALE = float(json.load(open(NORM_JSON))["scale"])
tf = json.load(open(DATA / "transforms.json"))
W, H, fx, fy, cx, cy = tf["w"], tf["h"], tf["fl_x"], tf["fl_y"], tf["cx"], tf["cy"]
UU, VV = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))

mesh = o3d.io.read_triangle_mesh(str(MESH))
smp = mesh.sample_points_uniformly(number_of_points=N_POINTS)
P = np.asarray(smp.points)
z_top = P[:, 2].max()

counts = np.zeros(len(P), dtype=np.int32)
n_frames = 0

for i, fr in enumerate(tf["frames"]):
    if SKIP_LOW_RING and i < 36:
        continue
    dp = GS / f"depth_{i:04d}.npy"
    ap = GS / f"acc_{i:04d}.npy"
    if not dp.exists():
        continue
    depth = np.load(dp).astype(np.float64) / SCALE
    acc = (np.load(ap).astype(np.float64) / 255.0) if ap.exists() else np.ones_like(depth)

    c2w = np.array(fr["transform_matrix"], dtype=np.float64)
    w2c = np.linalg.inv(c2w @ FLIP)
    R, t = w2c[:3, :3], w2c[:3, 3]

    # --- rebuild the exact v2 mask for this frame -------------------------
    valid = np.isfinite(depth) & (depth > 0.01) & (depth < DEPTH_TRUNC) & (acc > ACC_MIN)
    X = (UU - cx) * depth / fx
    Y = (VV - cy) * depth / fy
    pw = np.stack((X - t[0], Y - t[1], depth - t[2]), -1) @ R
    m = (valid
         & (np.abs(pw[..., 0]) < MASK_HALF_X) & (np.abs(pw[..., 1]) < MASK_HALF_Y)
         & (pw[..., 2] > TABLE_Z + TABLE_MARGIN) & (pw[..., 2] < MASK_Z_UP))
    if MASK_ERODE:
        m = binary_erosion(m, iterations=MASK_ERODE)
    n_frames += 1

    # --- project every sampled point into this frame ----------------------
    pc = P @ R.T + t                       # world -> camera
    Z = pc[:, 2]
    ok = Z > 0.01
    u = np.round(fx * pc[:, 0] / np.maximum(Z, 1e-9) + cx).astype(int)
    v = np.round(fy * pc[:, 1] / np.maximum(Z, 1e-9) + cy).astype(int)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.where(ok)[0]
    if idx.size == 0:
        continue
    hit = m[v[idx], u[idx]]
    agree = np.abs(depth[v[idx], u[idx]] - Z[idx]) < DEPTH_TOL
    counts[idx[hit & agree]] += 1

print(f"frames integrated: {n_frames}")
print(f"sampled surface points: {len(P)}")
print(f"\nobservations per point: mean {counts.mean():.1f}  median "
      f"{np.median(counts):.0f}  min {counts.min()}  max {counts.max()}")
for p in (1, 5, 10, 25, 50):
    print(f"  p{p:<3} = {np.percentile(counts, p):.0f}")
print(f"points NEVER observed: {(counts == 0).sum()} "
      f"({(counts == 0).mean()*100:.2f}%)")

print("\nby height band (relative to the mesh top):")
z = P[:, 2]
bands = [(0.000, 0.006, "top face"), (0.006, 0.015, "upper walls"),
         (0.015, 0.030, "lower walls"), (0.030, 0.999, "bottom / contact")]
for a, b, lbl in bands:
    sel = (z_top - z >= a) & (z_top - z < b)
    if sel.sum() < 5:
        continue
    print(f"  {lbl:<15} n={sel.sum():5d}  mean {counts[sel].mean():5.1f}  "
          f"median {np.median(counts[sel]):3.0f}  never {(counts[sel] == 0).sum():4d}")
