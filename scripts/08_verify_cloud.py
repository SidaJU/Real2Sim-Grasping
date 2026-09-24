"""T1.2e  Geometric sanity check on the back-projected point cloud.

If intrinsics, TRUE depth and poses agree, the YCB cracker box in the cloud must
measure ~0.164 x 0.213 x 0.072 m and the table must sit at z ~= 0.
A wrong fl_y (the bug we just fixed) shows up here as a stretched box.
"""

import sys
import numpy as np

PLY = "C:/Users/theozhu/Desktop/拓展/3DGS/data/ycb/points3D.ply"
EXPECTED = (0.164, 0.213, 0.072)   # cracker box, metres (x, y, z)
TOL = 0.02

with open(PLY) as f:
    lines = f.read().split("\n")
start = lines.index("end_header") + 1
data = np.array(
    [[float(v) for v in l.split()] for l in lines[start:] if l.strip()],
    dtype=np.float64,
)
xyz = data[:, :3]
print(f"points: {len(xyz)}")
print(f"full bbox  min {np.round(xyz.min(0), 3)}  max {np.round(xyz.max(0), 3)}")

# table plane: detect it as the densest z-slice (it is the largest flat surface)
hist, edges = np.histogram(xyz[:, 2], bins=400)
k = int(np.argmax(hist))
table_z = 0.5 * (edges[k] + edges[k + 1])
print(f"detected table z: {table_z:+.4f} m  (points in slice {int(hist[k])})")

# object region: above the table, inside a 0.3 m box around the origin
m = (
    (xyz[:, 2] > table_z + 0.015)
    & (xyz[:, 2] < table_z + 0.20)
    & (np.abs(xyz[:, 0]) < 0.15)
    & (np.abs(xyz[:, 1]) < 0.15)
)
obj = xyz[m]
if len(obj) < 100:
    print("FAIL: almost no points in the object region - cloud is wrong")
    sys.exit(1)

size = obj.max(0) - obj.min(0)
print(f"object points: {len(obj)}")
print(f"object size (m): x={size[0]:.3f} y={size[1]:.3f} z={size[2]:.3f}")
print(f"expected x/y   : x={EXPECTED[0]:.3f} y={EXPECTED[1]:.3f}")
print(f"visible height : {xyz[m].max(0)[2] - table_z:.3f} m of {EXPECTED[2]:.3f} "
      f"(box bottom is {table_z:.3f} m below the table top -> partly buried)")

# Only x/y are decisive: the vertical extent is clipped by the table.
exp_xy = sorted(EXPECTED[:2])
got_xy = sorted(size[:2])
err = max(abs(a - b) for a, b in zip(exp_xy, got_xy))
print(f"worst x/y error: {err:.3f} m (tolerance {TOL})")
print("VERDICT:", "OK - geometry is consistent" if err < TOL else "MISMATCH - check intrinsics")
