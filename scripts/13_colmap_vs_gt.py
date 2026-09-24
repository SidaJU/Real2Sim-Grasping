"""Quantify COLMAP pose accuracy against Isaac Sim ground truth.

Aligns the 50 COLMAP-estimated camera centers to the GT centers with a
similarity transform (Umeyama: scale + rotation + translation - COLMAP's
coordinate frame and scale are arbitrary), then reports the residual
per-camera position error. Convention-independent (uses camera centers only).
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\theozhu\miniconda3\envs\nerfstudio\lib\site-packages")
from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary

# usage: python 13_colmap_vs_gt.py [colmap_data_dir]
BASE = sys.argv[1] if len(sys.argv) > 1 else r"C:\ycb_sfm\gs"
COLMAP_MODEL = os.path.join(BASE, "sparse", "0", "images.bin")
GT_JSON = r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb\transforms.json"
print("model dir:", BASE)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def umeyama(src, dst):
    """Least-squares similarity transform src -> dst. Returns scale, R, t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    var_s = (s_c ** 2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - scale * R @ mu_s
    return scale, R, t


# --- COLMAP side: images.bin holds world-to-camera (q, t) ---
imgs = read_images_binary(COLMAP_MODEL)
colmap_c = {}
for v in imgs.values():
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(v.qvec)
    w2c[:3, 3] = v.tvec
    colmap_c[v.name] = (np.linalg.inv(w2c))[:3, 3]  # camera center in world

# --- GT side: transforms.json holds camera-to-world ---
gt = json.load(open(GT_JSON))
gt_c = {os.path.basename(f["file_path"]): np.array(f["transform_matrix"])[:3, 3]
        for f in gt["frames"]}

names = sorted(set(colmap_c) & set(gt_c))
print(f"matched frames: {len(names)} (of {len(colmap_c)} registered)")

src = np.array([colmap_c[n] for n in names])
dst = np.array([gt_c[n] for n in names])
scale, R, t = umeyama(src, dst)
print(f"scale factor COLMAP->GT: {scale:.4f}  (COLMAP units are arbitrary)")

aligned = (scale * (R @ src.T).T + t)
err = np.linalg.norm(aligned - dst, axis=1) * 100.0  # -> cm
rings = {0: "low 0.05m", 1: "mid 0.15m", 2: "top 0.30m"}
print("\nper-ring position error after alignment (cm):")
for r in range(3):
    e = np.array([err[i] for i, n in enumerate(names)
                  if int(n.split("_")[1].split(".")[0]) // 36 == r])
    if len(e):
        print(f"  ring{r} ({rings[r]}): n={len(e):2d}  mean {e.mean():5.2f}  max {e.max():5.2f}")
print(f"\nALL: mean {err.mean():.2f} cm   median {np.median(err):.2f} cm   "
      f"max {err.max():.2f} cm")

# ---- is it a few BAD images, or a systematic distortion of ALL of them? ----
# If only a handful were wrong, the median would stay low while the mean is
# dragged up. If median ~ mean, every single camera is off -> systemic.
print("\nerror distribution (cm):")
for q in (10, 25, 50, 75, 90):
    print(f"  p{q:<2d}: {np.percentile(err, q):6.2f}")
print(f"  p50/p90 ratio: {np.median(err) / np.percentile(err, 90):.2f}  "
      f"(near 1.0 => every camera is off, NOT a few bad frames)")

print("\nworst 8 frames:")
order = np.argsort(-err)
for i in order[:8]:
    print(f"  {names[i]}  {err[i]:6.2f} cm")
print("best 3 frames:")
for i in order[-3:]:
    print(f"  {names[i]}  {err[i]:6.2f} cm")

# how many frames would we have to drop to get under 5 cm?
good = (err < 5).sum()
print(f"\nframes under 5 cm error: {good}/{len(err)} "
      f"({100.0 * good / len(err):.0f}%)  -> dropping 'bad' frames "
      f"{'is viable' if good > len(err) * 0.7 else 'cannot fix this'}")

# spread of camera heights in COLMAP's own frame (GT: each ring is a flat circle)
z = src[:, 2]
print(f"\nCOLMAP-frame z of all cameras: min {z.min():.2f} max {z.max():.2f} "
      f"(arbitrary orientation -> not comparable to real heights)")
