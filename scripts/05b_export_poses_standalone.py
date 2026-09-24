"""T1.2b-2 (standalone)  Write nerfstudio transforms.json WITHOUT Isaac Sim running.

Same math as 05_export_poses.py, but the camera intrinsics are explicit instead of
being read off the live stage (Try.usd is binary .usdc, so we cannot grep them).

  focal 10.0 mm, aperture 20.955 mm  <- exactly what 04_render_orbit.py sets up

Also copies the images into the nerfstudio data layout:
  3DGS/data/ycb/images/*.png
  3DGS/data/ycb/transforms.json
"""

import os
import math
import json
import shutil
import numpy as np

SRC = "C:/isaacsim_data/ycb_v2"                              # 04_render_orbit.py output
DST = "C:/Users/theozhu/Desktop/拓展/3DGS/data/ycb"          # nerfstudio data root
NUM, RADIUS, HEIGHT = 36, 0.25, 0.15                         # MUST match 04
TARGET = (0, 0, 0.036)
RES = (640, 640)
FOCAL_MM = 10.0          # 04 sets this on the camera it creates
APERTURE_MM = 20.955     # Kit default horizontal aperture

pngs = sorted([f for f in os.listdir(SRC) if f.endswith(".png")])
print("images found:", len(pngs))
if len(pngs) < NUM:
    print(f"WARNING: expected >= {NUM} images, got {len(pngs)}")


def look_at_c2w(eye, target, up=(0, 0, 1)):
    """USD convention: camera looks down -Z, world is Z-up. Returns 4x4 c2w."""
    eye, target, up = map(np.array, (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd                                    # camera -Z is forward
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


FLIP = np.diag([1.0, -1.0, -1.0, 1.0])         # USD(OpenGL) -> OpenCV(nerfstudio)
fl_px = FOCAL_MM / APERTURE_MM * RES[0]

positions = [
    (
        RADIUS * math.cos(2 * math.pi * i / NUM),
        RADIUS * math.sin(2 * math.pi * i / NUM),
        HEIGHT,
    )
    for i in range(NUM)
]

frames = []
for i, eye in enumerate(positions[: min(NUM, len(pngs))]):
    c2w = look_at_c2w(eye, TARGET) @ FLIP
    frames.append(
        {"file_path": f"images/{pngs[i]}", "transform_matrix": c2w.tolist()}
    )

payload = {
    "camera_model": "OPENCV",
    "fl_x": fl_px,
    "fl_y": fl_px,
    "cx": RES[0] / 2,
    "cy": RES[1] / 2,
    "w": RES[0],
    "h": RES[1],
    "frames": frames,
}

os.makedirs(DST + "/images", exist_ok=True)
with open(DST + "/transforms.json", "w") as f:
    json.dump(payload, f, indent=2)
with open(SRC + "/transforms.json", "w") as f:      # keep a copy next to the renders
    json.dump(payload, f, indent=2)

for name in pngs[: min(NUM, len(pngs))]:
    shutil.copy2(os.path.join(SRC, name), os.path.join(DST, "images", name))

print(f"transforms.json: {len(frames)} frames")
print(f"fl_x = fl_y = {fl_px:.2f} px   size {RES[0]}x{RES[1]}")
print("data written to:", DST)
