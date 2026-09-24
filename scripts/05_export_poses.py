"""T1.2b-2  Collect rendered PNGs and write nerfstudio transforms.json.

Run this ~30-60s AFTER 04_render_orbit.py (check the printed count first).
Camera poses are recomputed from the same deterministic orbit as 04, so the
two scripts stay consistent.
"""

import os
import math
import json
import numpy as np

OUTPUT = "C:/isaacsim_data/ycb_v2"   # keep in sync with 04_render_orbit.py
NUM, RADIUS, HEIGHT = 36, 0.25, 0.15   # MUST stay in sync with 04_render_orbit.py
TARGET = (0, 0, 0.036)
RES = (640, 640)

# read the ACTUAL optics off the camera prim instead of hardcoding, so the
# intrinsics here always match what 04 really rendered
import omni.usd
from pxr import UsdGeom
_cam = UsdGeom.Camera(
    omni.usd.get_context().get_stage().GetPrimAtPath("/World/OrbitCamera")
)
FOCAL_MM = _cam.GetFocalLengthAttr().Get() or 50.0
APERTURE_MM = _cam.GetHorizontalApertureAttr().Get() or 20.955
print("camera focal:", FOCAL_MM, "aperture:", APERTURE_MM)

pngs = sorted([f for f in os.listdir(OUTPUT) if f.endswith(".png")])
print("rendered:", len(pngs))
if not pngs:
    print("no images yet - wait longer, or check that 04 ran without errors")
    raise SystemExit


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
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye   # columns = camera axes
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
for i, eye in enumerate(positions[: len(pngs)]):
    c2w = look_at_c2w(eye, TARGET) @ FLIP
    frames.append(
        {"file_path": f"images/{pngs[i]}", "transform_matrix": c2w.tolist()}
    )

with open(OUTPUT + "/transforms.json", "w") as f:
    json.dump(
        {
            "camera_model": "OPENCV",
            "fl_x": fl_px,
            "fl_y": fl_px,
            "cx": RES[0] / 2,
            "cy": RES[1] / 2,
            "w": RES[0],
            "h": RES[1],
            "frames": frames,
        },
        f,
        indent=2,
    )
print("transforms.json:", len(frames), "frames")
