"""T6b  Keep only points inside the object's bounding box in points3D.ply.

The initial point cloud was back-projected from ALL pixels, so it contains
table-top "dust" from depth noise. Those points became gaussians at init;
after masking, background pixels have no gradient, so the bad gaussians can
never be pulled in or culled - they sit there forever as red/white debris.

Fix: keep only points inside the GT mesh bounding box (slightly padded).
"""

from pathlib import Path

import numpy as np
import open3d as o3d

PTS = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb\points3D.ply")
GT_MESH = Path(r"C:\isaacsim_data\mesh\gt_mesh.obj")
PAD = 0.012

gt = o3d.io.read_triangle_mesh(str(GT_MESH))
v = np.asarray(gt.vertices)
lo, hi = v.min(0) - PAD, v.max(0) + PAD
print(f"object bbox: {np.round(lo,3)} .. {np.round(hi,3)}")

pcd = o3d.io.read_point_cloud(str(PTS))
pts = np.asarray(pcd.points)
print(f"points before: {len(pts)}")

m = (
    (pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0])
    & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1])
    & (pts[:, 2] > lo[2]) & (pts[:, 2] < hi[2])
)
kept = pcd.select_by_index(np.where(m)[0])
print(f"points after : {len(kept.points)}  (dropped {len(pts) - len(kept.points)})")

backup = PTS.with_suffix(".full.ply")
if not backup.exists():
    o3d.io.write_point_cloud(str(backup), pcd)
    print("backup of the full cloud saved:", backup)

o3d.io.write_point_cloud(str(PTS), kept)
print("filtered cloud written:", PTS)
