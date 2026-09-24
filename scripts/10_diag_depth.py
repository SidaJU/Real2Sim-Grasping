"""T1.2g  Unambiguous test: which fy keeps the TABLE horizontal?

Take pixels from two rows that both hit the table (lower part of the frame,
same column). Back-project with a candidate fy. Because the table is horizontal,
both must land at the SAME world height. A wrong fy tilts the table, so the two
heights differ. This needs no assumption about where the table actually is.

Also checks the centre pixel against the analytic ray-box distance, which is
independent of fy (centre pixel has u=cx, v=cy).
"""

import json
import math
import numpy as np

SRC = "C:/isaacsim_data/ycb_v3"
NUM_VIEWS, RADIUS = 36, 0.25
HEIGHTS = [0.05, 0.15, 0.30]
IDX = 72                 # height 0.30 -> looks down, table fills lower frame
ROWS = (430, 520, 600)   # rows that hit the table
COL = 320

intr = json.load(open(SRC + "/intrinsics.json"))
W, H = intr["res"]
fx = intr["focal_mm"] / intr["h_aperture_mm"] * W
cx, cy = W / 2.0, H / 2.0
fy_a = fx
fy_b = intr["focal_mm"] / intr["v_aperture_mm"] * H


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
eye = np.array(
    [
        RADIUS * math.cos(2 * math.pi * vi / NUM_VIEWS),
        RADIUS * math.sin(2 * math.pi * vi / NUM_VIEWS),
        h,
    ]
)
c2w = look_at_c2w(eye) @ FLIP
depth = np.load(f"{SRC}/depth_{IDX:04d}.npy").reshape(H, W)

print(f"frame {IDX}: eye={np.round(eye, 3)}  height={h}")

# --- 1. centre pixel vs analytic ray/box distance (fy-independent) ---
half = np.array([0.164, 0.213, 0.072]) / 2.0          # box half extents
centre = np.array([0.0, 0.0, 0.036])                  # box centre
fwd = (centre - eye) / np.linalg.norm(centre - eye)
tmin, tmax = 0.0, 1e9
for a in range(3):                                     # slab test
    if abs(fwd[a]) < 1e-9:
        continue
    t1 = (centre[a] - half[a] - eye[a]) / fwd[a]
    t2 = (centre[a] + half[a] - eye[a]) / fwd[a]
    tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
print(f"\ncentre pixel: analytic box hit at {tmin:.4f} m | measured depth "
      f"{depth[320, 320]:.4f} m")

# --- 2. table horizontality test ---
print("\ntable rows -> world height (correct fy gives EQUAL heights)")
for name, fy in (("A fy=fx=%.1f" % fx, fy_a), ("B fy=%.1f" % fy_b, fy_b)):
    heights = []
    for v in ROWS:
        d = depth[v, COL]
        if not np.isfinite(d) or d <= 0:
            heights.append(float("nan"))
            continue
        pc = np.array([(COL - cx) * d / fx, (v - cy) * d / fy, d, 1.0])
        pw = c2w @ pc
        heights.append(pw[2])
    hs = np.array(heights)
    spread = np.nanmax(hs) - np.nanmin(hs)
    print(f"  {name}: heights={np.round(hs, 4)}  spread={spread * 1000:.1f} mm")
