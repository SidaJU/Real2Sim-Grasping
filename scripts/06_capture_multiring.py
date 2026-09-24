"""T1.2c  Multi-height orbit capture with TRUE depth + real intrinsics.

Why this replaces 04_render_orbit.py:
  * 3 heights (0.05/0.15/0.30) x 36 views = 108 frames -> top face finally covered
  * captures distance_to_image_plane (TRUE depth) alongside rgb
  * saves the camera's REAL focal/aperture -> no more hardcoded intrinsics

Output (per frame):  rgb_XXXX.npy (H,W,4 uint8) + depth_XXXX.npy (H,W float32, meters)
Plus intrinsics.json.  07_build_init.py turns these into the nerfstudio dataset.

Rules kept from 04 (all learned the hard way):
  * timeline must be RUNNING for capture -> we play/stop inside the coroutine
  * cancel the previous coroutine first (Script Editor keeps globals)
  * annotators are singletons -> detach stale before attach
  * lighting/exposure are NOT touched here
"""

import omni.replicator.core as rep
import omni.usd
import omni.timeline
import omni.kit.async_engine
import math
import os
import json
import numpy as np
from pxr import UsdGeom, Gf

OUTPUT = "C:/isaacsim_data/ycb_v3"
NUM_VIEWS, RADIUS = 36, 0.25
HEIGHTS = [0.05, 0.15, 0.30]
TARGET = np.array([0.0, 0.0, 0.036])
RES = (640, 640)
CAM_PATH = "/World/OrbitCamera"

stage = omni.usd.get_context().get_stage()

# ---- clean stale capture ----
os.makedirs(OUTPUT, exist_ok=True)
for f in os.listdir(OUTPUT):
    os.remove(os.path.join(OUTPUT, f))

# ---- annotators: rgb + true depth (singletons -> detach stale first) ----
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
for a in (rgb_annot, depth_annot):
    try:
        a.detach()
    except Exception:
        pass

rp = rep.create.render_product(CAM_PATH, RES)
rgb_annot.attach([rp])
depth_annot.attach([rp])

# ---- save REAL intrinsics off the prim ----
cam = UsdGeom.Camera(stage.GetPrimAtPath(CAM_PATH))
intr = {
    "focal_mm": cam.GetFocalLengthAttr().Get(),
    "h_aperture_mm": cam.GetHorizontalApertureAttr().Get(),
    "v_aperture_mm": cam.GetVerticalApertureAttr().Get(),
    "res": list(RES),
}
with open(OUTPUT + "/intrinsics.json", "w") as f:
    json.dump(intr, f, indent=2)
print("intrinsics:", intr)


def look_at_matrix(eye, target, up=np.array([0.0, 0.0, 1.0])):
    eye = np.asarray(eye, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd                                   # USD camera looks down -Z
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def set_cam_pose(eye):
    m = look_at_matrix(eye, TARGET)
    xfo = UsdGeom.Xformable(stage.GetPrimAtPath(CAM_PATH))
    xfo.ClearXformOpOrder()
    xfo.AddTransformOp().Set(Gf.Matrix4d(*m.T.flatten().tolist()))


# ---- cancel a still-running previous coroutine ----
try:
    if _render_task is not None and not _render_task.done():
        _render_task.cancel()
        print("cancelled previous render task")
except NameError:
    pass


async def render_all():
    tl = omni.timeline.get_timeline_interface()
    tl.stop()
    tl.play()
    # Warm-up: auto-exposure needs a few frames to settle. Measured drift was
    # 214->208 brightness over the first 5 frames -> ghost gaussians later.
    for _ in range(20):
        await rep.orchestrator.step_async()
    idx = 0
    for h in HEIGHTS:
        for i in range(NUM_VIEWS):
            eye = (
                RADIUS * math.cos(2 * math.pi * i / NUM_VIEWS),
                RADIUS * math.sin(2 * math.pi * i / NUM_VIEWS),
                h,
            )
            set_cam_pose(eye)
            await rep.orchestrator.step_async()
            np.save(f"{OUTPUT}/rgb_{idx:04d}.npy", rgb_annot.get_data())
            np.save(f"{OUTPUT}/depth_{idx:04d}.npy", depth_annot.get_data())
            idx += 1
    tl.stop()
    print(f"DONE - {idx} frames (rgb+depth npy) in {OUTPUT} - now run 07_build_init.py")


_render_task = omni.kit.async_engine.run_coroutine(render_all())
print("capturing started (async, 108 frames, ~1-2 min)")
