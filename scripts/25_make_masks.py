"""T6  Generate object masks from TRUE depth (no re-rendering needed).

Idea: we know exactly where the object is (GT mesh bbox) and we have per-pixel
true depth + poses. Back-project every pixel to world space; anything landing
inside the object's bounding box is the object, everything else is background
that we mask OUT.

Why: the pure white background gives 3DGS zero multi-view constraint, so the
gaussians drift outside the camera sphere (background collapse). Training with
a mask forces all gaussians to stay on the object.

Output: data/ycb/masks/*.png (255 = keep, 0 = ignore)
        transforms.json gains a "mask_path" per frame (nerfstudio requires it
        on every frame or none)
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

DATA = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb")
GT_MESH = Path(r"C:\isaacsim_data\mesh\gt_mesh.obj")     # ASCII path
DEPTH = DATA / "depth"
MASKS = DATA / "masks"
PAD = 0.006          # dilate the box slightly so edges are not clipped

# ---- object bounds in world metres (from the GT mesh) ----
gt = o3d.io.read_triangle_mesh(str(GT_MESH))
v = np.asarray(gt.vertices)
lo, hi = v.min(0) - PAD, v.max(0) + PAD
print(f"object bbox: {np.round(lo,3)} .. {np.round(hi,3)}")

tf = json.load(open(DATA / "transforms.json"))
W, H, fx, fy, cx, cy = tf["w"], tf["h"], tf["fl_x"], tf["fl_y"], tf["cx"], tf["cy"]

MASKS.mkdir(exist_ok=True)
for f in MASKS.glob("*.png"):
    f.unlink()

us, vs = np.meshgrid(np.arange(W), np.arange(H))      # (H,W)
ones = np.ones_like(us, dtype=np.float32)

coverages = []
for i, fr in enumerate(tf["frames"]):
    dep_p = DEPTH / f"rgb_{i:04d}.npy"
    if not dep_p.exists():
        print(f"missing depth for frame {i}")
        continue
    depth = np.load(dep_p).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0.01) & (depth < 1.0)

    # camera-space -> world (OpenCV convention, same as transforms.json)
    x_c = (us - cx) * depth / fx
    y_c = (vs - cy) * depth / fy
    pts_c = np.stack([x_c, y_c, depth, ones], axis=-1)      # (H,W,4)
    c2w = np.array(fr["transform_matrix"], dtype=np.float32)
    pts_w = pts_c @ c2w.T                                    # (H,W,4)
    xyz = pts_w[..., :3]

    inside = (
        (xyz[..., 0] > lo[0]) & (xyz[..., 0] < hi[0])
        & (xyz[..., 1] > lo[1]) & (xyz[..., 1] < hi[1])
        & (xyz[..., 2] > lo[2]) & (xyz[..., 2] < hi[2])
        & valid
    )

    # nerfstudio semantics: mask = 1 means "EXCLUDE this pixel" (originally for
    # dynamic objects). We want to KEEP the object, so the BACKGROUND must be
    # 255 and the object 0.
    mask = (~inside * 255).astype(np.uint8)

    # clean up speckles
    m = o3d.geometry.Image(mask)
    # (open3d image filters need a mesh-free path; use numpy instead)
    from scipy.ndimage import binary_closing, binary_opening
    b = binary_closing(inside, np.ones((5, 5)))
    b = binary_opening(b, np.ones((3, 3)))
    mask = (~b * 255).astype(np.uint8)          # 255 = background = excluded

    name = Path(fr["file_path"]).name
    Image.fromarray(mask).save(MASKS / name)
    coverages.append(float(b.mean()))
    if i % 20 == 0:
        print(f"frame {i:3d}: object covers {b.mean()*100:5.1f}% of the frame")

# ---- write mask_path into transforms.json ----
for fr in tf["frames"]:
    fr["mask_path"] = "masks/" + Path(fr["file_path"]).name
with open(DATA / "transforms.json", "w") as f:
    json.dump(tf, f, indent=2)

print(f"\nmasks written: {len(list(MASKS.glob('*.png')))}")
print(f"mean object coverage: {np.mean(coverages)*100:.1f}% of each frame")
print("transforms.json updated with mask_path - retrain to see the gaussians")
print("cluster on the object instead of drifting outside the camera sphere")
