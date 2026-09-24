"""T1.2f  Decide the true fy using the box thickness as ground truth.

The table and the box top are two parallel planes. Whatever absolute z they land
on, their SEPARATION must equal the cracker box thickness (0.072 m).
A wrong fy stretches z, so the measured gap will not be 0.072.

Candidates:
  A) fy = fx          -> isotropic pixels (Kit ignores vertical aperture)
  B) fy from v_aperture -> anisotropic pixels (USD aperture semantics)
"""

import json
import math
import numpy as np

SRC = "C:/isaacsim_data/ycb_v3"
NUM_VIEWS, RADIUS = 36, 0.25
HEIGHTS = [0.05, 0.15, 0.30]
TARGET_Z = 0.036
IDX = 72          # height 0.30 -> looks DOWN on the box: sees box top AND table
TRUE_THICKNESS = 0.072

intr = json.load(open(SRC + "/intrinsics.json"))
W, H = intr["res"]
fx = intr["focal_mm"] / intr["h_aperture_mm"] * W
cx, cy = W / 2.0, H / 2.0
fy_a = fx
fy_b = intr["focal_mm"] / intr["v_aperture_mm"] * H
print(f"fx={fx:.1f}  fy_A={fy_a:.1f}  fy_B={fy_b:.1f}\n")


def look_at_c2w(eye, target=(0, 0, 0.036), up=(0, 0, 1)):
    eye, target, up = map(np.array, (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
hi, vi = divmod(IDX, NUM_VIEWS)
h = HEIGHTS[hi]
eye = (
    RADIUS * math.cos(2 * math.pi * vi / NUM_VIEWS),
    RADIUS * math.sin(2 * math.pi * vi / NUM_VIEWS),
    h,
)
c2w = look_at_c2w(eye) @ FLIP
depth = np.load(f"{SRC}/depth_{IDX:04d}.npy").reshape(H, W)

vs, us = np.mgrid[0:H:2, 0:W:2]
z = depth[::2, ::2]
valid = np.isfinite(z) & (z > 0.01) & (z < 1.0)
print(f"valid depth pixels: {valid.sum()} / {z.size}")

for name, fy in (("A isotropic (fy=fx)", fy_a), ("B v_aperture", fy_b)):
    x_ = (us[valid] - cx) * z[valid] / fx
    y_ = (vs[valid] - cy) * z[valid] / fy
    pc = np.stack([x_, y_, z[valid], np.ones_like(z[valid])], 0)
    pw = c2w @ pc
    zw = pw[2]

    hist, edges = np.histogram(zw, bins=400)
    peaks = []
    for k in range(2, len(hist) - 2):
        if hist[k] > hist[k - 1] and hist[k] >= hist[k + 1] and hist[k] > 0.03 * hist.max():
            peaks.append((0.5 * (edges[k] + edges[k + 1]), int(hist[k])))
    peaks.sort(key=lambda p: -p[1])
    peaks = peaks[:4]
    peaks.sort()
    print(f"\n{name}")
    print("  z layers (height, points): " + "  ".join(f"{p:+.4f}({n})" for p, n in peaks))
    if len(peaks) >= 2:
        gaps = [round(b[0] - a[0], 4) for a, b in zip(peaks, peaks[1:])]
        print(f"  gaps: {gaps}   -> thickness should be {TRUE_THICKNESS}")
        best = min(gaps, key=lambda g: abs(g - TRUE_THICKNESS))
        print(f"  closest gap: {best:.4f}  error {abs(best - TRUE_THICKNESS) * 1000:.1f} mm")
