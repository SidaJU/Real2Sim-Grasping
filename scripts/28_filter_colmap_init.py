"""Clean background / junk points out of a COLMAP sparse point cloud.

Lowering SiftExtraction.peak_threshold to rescue registration rate also makes
COLMAP pick up noise features on the plain white table, so ~22% of the init
points end up far away from the object and ~37% are gray (table). Those seeds
get multiplied by densification and show up as diffuse gray-white gaussians.

Two filters, both ground-truth-free (a real pipeline can do exactly this):
  1. track length  - a point seen by < MIN_TRACK images is usually a bad match
  2. statistical outlier removal (open3d) - drops isolated floaters

Writes the result back as points3D.bin (nerfstudio reads it for init).
The original is backed up to points3D.bin.orig.
"""
import shutil
import struct
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\theozhu\miniconda3\envs\nerfstudio\lib\site-packages")
from nerfstudio.data.utils.colmap_parsing_utils import read_points3D_binary

SPARSE = r"C:\ycb_sfm\gs\sparse\0"
BIN = SPARSE + r"\points3D.bin"
MIN_TRACK = 3           # keep points observed by at least this many images
OUTLIER_NB = 16         # open3d remove_statistical_outlier neighbours
OUTLIER_STD = 2.0       # std ratio (smaller = more aggressive)

pts = read_points3D_binary(BIN)
print(f"input points: {len(pts)}")

ids = np.array(sorted(pts.keys()))
xyz = np.array([pts[i].xyz for i in ids])
rgb = np.array([pts[i].rgb for i in ids], dtype=np.uint8)
err = np.array([pts[i].error for i in ids])
tracks = [pts[i].image_ids for i in ids]
p2d = [pts[i].point2D_idxs for i in ids]

# ---- filter 1: track length ----
tlen = np.array([len(t) for t in tracks])
keep = tlen >= MIN_TRACK
print(f"track >= {MIN_TRACK}: keep {keep.sum()} / {len(keep)} "
      f"(dropped {len(keep) - keep.sum()})")
xyz, rgb, err, tracks, p2d = xyz[keep], rgb[keep], err[keep], \
    [t for t, k in zip(tracks, keep) if k], [t for t, k in zip(p2d, keep) if k]

# ---- filter 2: statistical outlier removal ----
try:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, inl = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NB,
                                            std_ratio=OUTLIER_STD)
    inl = np.array(inl)
    print(f"outlier removal (nb={OUTLIER_NB}, std={OUTLIER_STD}): "
          f"keep {len(inl)} / {len(xyz)} (dropped {len(xyz) - len(inl)})")
    xyz, rgb, err = xyz[inl], rgb[inl], err[inl]
    tracks = [tracks[i] for i in inl]
    p2d = [p2d[i] for i in inl]
except ImportError:
    print("open3d not available - skipping outlier removal")

print(f"final points: {len(xyz)}")

# ---- write back as COLMAP binary ----
shutil.copy(BIN, BIN + ".orig")
with open(BIN, "wb") as f:
    f.write(struct.pack("<Q", len(xyz)))
    for i in range(len(xyz)):
        f.write(struct.pack("<Q", i + 1))
        f.write(struct.pack("<ddd", *xyz[i]))
        f.write(struct.pack("<BBB", *rgb[i]))
        f.write(struct.pack("<d", err[i]))
        f.write(struct.pack("<Q", len(tracks[i])))
        for img_id, p2 in zip(tracks[i], p2d[i]):
            f.write(struct.pack("<ii", int(img_id), int(p2)))

print(f"written: {BIN}   (backup: {BIN}.orig)")
print("now retrain with run_train_colmap.bat")
