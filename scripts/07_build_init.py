"""T1.2d (standalone)  npy capture -> nerfstudio dataset + true-depth init cloud.

Input : C:/isaacsim_data/ycb_v3  (rgb_XXXX.npy, depth_XXXX.npy, intrinsics.json)
Output: 3DGS/data/ycb/
          images/rgb_XXXX.png        8-bit PNGs
          transforms.json            REAL intrinsics + ply_file_path field
          points3D.ply               back-projected TRUE depth (USD world coords)

The ply_file_path key is what makes nerfstudio's NerfstudioDataParser load the
cloud (instead of random initialization). Points are in USD world meters; the
parser applies the same normalization to them as to the camera poses.

Back-projection (OpenCV camera frame, matches the FLIP'd c2w):
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z                (distance_to_image_plane = depth along view axis)
    p_world = c2w_opencv @ p_cam
"""

import os
import json
import math
import numpy as np
from PIL import Image

SRC = "C:/isaacsim_data/ycb_v3"
DST = "C:/Users/theozhu/Desktop/拓展/3DGS/data/ycb"
NUM_VIEWS, RADIUS = 36, 0.25
HEIGHTS = [0.05, 0.15, 0.30]
TARGET = (0, 0, 0.036)
MAX_PTS = 150000
STRIDE = 2          # take every 2nd pixel
Z_MIN, Z_MAX = 0.01, 0.6   # object + nearby table only (far table breeds floaters)
XY_MAX = 0.35              # keep points within 35cm of the object (table extends far)
BRIGHT_LIMIT = 212          # drop frames captured before auto-exposure settled
NORMALIZE_EXPOSURE = True   # equalise per-frame gain (linear space)
COPY_DEPTH = True           # also drop the true depth next to the images

intr = json.load(open(SRC + "/intrinsics.json"))
W, H = intr["res"]
fx = intr["focal_mm"] / intr["h_aperture_mm"] * W
# Kit renders ISOTROPICALLY: verticalAperture does NOT affect the image, so
# fy == fx. Verified by 10_diag_depth.py - using v_aperture tilted the table by
# 15 mm across the frame, while fy=fx kept it flat to 0.1 mm.
fy = fx
cx, cy = W / 2.0, H / 2.0
print(f"intrinsics: fx=fy={fx:.2f}px  cx={cx} cy={cy}  ({W}x{H})")

pngs_all = sorted(f for f in os.listdir(SRC) if f.startswith("rgb_") and f.endswith(".npy"))
# drop frames with abnormal exposure (captured before auto-exposure settled)
pngs = []
dropped = []
for fn in pngs_all:
    rgb = np.load(os.path.join(SRC, fn))
    if rgb[..., :3].mean() > BRIGHT_LIMIT:
        dropped.append(fn)
    else:
        pngs.append(fn)
print(f"frames: {len(pngs)} kept, {len(dropped)} dropped for exposure:")
for d in dropped:
    print("  dropped", d)
N = len(pngs)
if N == 0:
    raise SystemExit("no capture found - run 06_capture_multiring.py in Isaac Sim first")


def look_at_c2w(eye, target, up=(0, 0, 1)):
    """USD convention: camera looks down -Z, world is Z-up. Returns 4x4 c2w."""
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


# IMPORTANT: do NOT convert USD poses to "OpenCV" convention.
# nerfstudio's cameras look down -Z (OpenGL/Blender convention), which is
# exactly what look_at_c2w() already produces for USD. Applying the old
# diag(1,-1,-1) FLIP turned every camera around by 180 degrees, so the gaussians
# ended up BEHIND the cameras: accumulation == 0, zero gradient, no densification
# and the model stayed a frozen translucent point cloud (debugged 2026-09-22).
# Keep c2w as-is; only the back-projection below is expressed in OpenCV camera
# axes and is flipped internally there.
FLIP = np.eye(4)   # no-op: USD/OpenGL c2w is what nerfstudio expects

# ---- exposure normalisation ----------------------------------------------
# Auto-exposure follows frame content, so brightness drifts with view angle.
# 3DGS assumes one colour per surface across ALL views, so we equalise the
# per-frame gain in LINEAR space (equivalent to shooting with fixed exposure).
def linear_mean(fn):
    a = np.load(os.path.join(SRC, fn))[..., :3].astype(np.float32) / 255.0
    return float(np.power(a, 2.2).mean())


if NORMALIZE_EXPOSURE:
    means = [linear_mean(fn) for fn in pngs]
    EXP_TARGET = float(np.median(means))
    print(f"exposure: linear means {min(means):.4f}..{max(means):.4f} "
          f"-> normalising all frames to {EXP_TARGET:.4f}")
else:
    EXP_TARGET = None

img_dir = os.path.join(DST, "images")
depth_dir = os.path.join(DST, "depth")
os.makedirs(img_dir, exist_ok=True)
if COPY_DEPTH:
    os.makedirs(depth_dir, exist_ok=True)
    for f in os.listdir(depth_dir):
        os.remove(os.path.join(depth_dir, f))
for f in os.listdir(img_dir):
    if f.endswith(".png"):
        os.remove(os.path.join(img_dir, f))

frames = []
pts_all, col_all = [], []
for idx, fn in enumerate(pngs):
    # map back to the ORIGINAL capture index - dropped frames must not shift
    # the pose assignment of the kept ones
    orig_idx = int(fn.split("_")[1].split(".")[0])
    hi, vi = divmod(orig_idx, NUM_VIEWS)
    h = HEIGHTS[hi]
    eye = (
        RADIUS * math.cos(2 * math.pi * vi / NUM_VIEWS),
        RADIUS * math.sin(2 * math.pi * vi / NUM_VIEWS),
        h,
    )
    c2w_cv = look_at_c2w(eye, TARGET) @ FLIP   # OpenCV convention (y down, z fwd)

    rgb = np.load(os.path.join(SRC, fn))       # H,W,4 uint8
    depth = np.load(os.path.join(SRC, fn.replace("rgb_", "depth_")))
    depth = depth.reshape(H, W)

    if EXP_TARGET is not None:
        lin = np.power(rgb[..., :3].astype(np.float32) / 255.0, 2.2)
        cur = lin.mean()
        if cur > 1e-6:
            lin = np.clip(lin * (EXP_TARGET / cur), 0.0, 1.0)
        rgb = np.power(lin, 1.0 / 2.2) * 255.0
        rgb = rgb.clip(0, 255).astype(np.uint8)
    else:
        rgb = rgb[:, :, :3]

    Image.fromarray(rgb[:, :, :3]).save(os.path.join(img_dir, f"rgb_{idx:04d}.png"))

    # keep the TRUE depth next to the images so TSDF fusion (15_*) can find it
    if COPY_DEPTH:
        np.save(os.path.join(depth_dir, f"rgb_{idx:04d}.npy"), depth)
    frames.append(
        {"file_path": f"images/rgb_{idx:04d}.png", "transform_matrix": c2w_cv.tolist()}
    )

    # back-project every STRIDE-th pixel with a valid depth
    vs, us = np.mgrid[0:H:STRIDE, 0:W:STRIDE]
    z = depth[::STRIDE, ::STRIDE]
    valid = np.isfinite(z) & (z > Z_MIN) & (z < Z_MAX)
    if valid.sum() == 0:
        continue
    x_ = (us[valid] - cx) * z[valid] / fx
    y_ = (vs[valid] - cy) * z[valid] / fy
    # (x_, y_, z) is in OpenCV camera axes (+X right, +Y down, +Z forward).
    # c2w_cv is now OpenGL (+X right, +Y up, -Z forward), so flip Y and Z to
    # express the same ray in the camera frame c2w_cv actually uses.
    pc = np.stack([x_, -y_, -z[valid], np.ones_like(z[valid])], axis=0)  # 4,M
    pw = c2w_cv @ pc                                                   # 4,M
    pts_all.append(pw[:3].T)
    col_all.append(rgb[::STRIDE, ::STRIDE, :3][valid].reshape(-1, 3))

pts = np.concatenate(pts_all)
cols = np.concatenate(col_all)
# drop far table points: they add nothing to the object and later turn into
# background floaters once densification starts multiplying them.
keep = np.linalg.norm(pts[:, :2], axis=1) < XY_MAX
pts, cols = pts[keep], cols[keep]
if len(pts) > MAX_PTS:
    sel = np.random.RandomState(0).choice(len(pts), MAX_PTS, replace=False)
    pts, cols = pts[sel], cols[sel]
print("init point cloud:", len(pts), "points")

ply_path = os.path.join(DST, "points3D.ply")
with open(ply_path, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(pts)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")
    for (x, y, z), (r, g, b) in zip(pts.tolist(), cols.tolist()):
        f.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")

payload = {
    "camera_model": "OPENCV",
    "fl_x": fx,
    "fl_y": fy,
    "cx": cx,
    "cy": cy,
    "w": W,
    "h": H,
    "ply_file_path": "points3D.ply",     # <- makes nerfstudio load the cloud
    "frames": frames,
}
with open(os.path.join(DST, "transforms.json"), "w") as f:
    json.dump(payload, f, indent=2)

print(f"transforms.json: {len(frames)} frames, fl_x={fx:.2f}")
print("dataset ready:", DST)
