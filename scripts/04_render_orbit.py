"""T1.2b  GUI orbit render - re-runnable in Script Editor.

Rules learned the hard way:
  * Replicator writers only capture while the timeline is RUNNING -> we start it.
  * The writer is a global singleton -> detach stale products before attaching.
  * The orbit camera is REUSED if it exists, so all optics you tuned in the
    editor (fStop, focal, clipping, exposure) are preserved.
  * A still-running previous coroutine must be cancelled first, otherwise two
    coroutines fight over the camera transform.
  * Lighting/exposure are NOT touched here - set them in the editor.
"""

import omni.replicator.core as rep
import omni.usd
import omni.timeline
import omni.kit.async_engine
import math
import os
import numpy as np
from pxr import UsdGeom, Gf

OUTPUT = "C:/isaacsim_data/ycb_v2"       # keep in sync with 05_export_poses.py
NUM_VIEWS, RADIUS, HEIGHT = 36, 0.25, 0.15
TARGET = np.array([0.0, 0.0, 0.036])
RES = (640, 640)
CAM_PATH = "/World/OrbitCamera"

stage = omni.usd.get_context().get_stage()

# ---- clean stale images ----
os.makedirs(OUTPUT, exist_ok=True)
for f in os.listdir(OUTPUT):
    if f.endswith(".png"):
        os.remove(os.path.join(OUTPUT, f))

# ---- writer: drop stale attachments, then attach ONE render product ----
writer = rep.WriterRegistry.get("BasicWriter")
rep.orchestrator.stop()          # kill any queued captures from stale coroutines
try:
    writer.detach()
except Exception:
    pass
# A render_product left over from a previous run KEEPS WRITING on every step.
# Left unchecked you get 3 products -> 3 images per step (123 instead of 41).
_old_rp = globals().get("_rp")
if _old_rp is not None:
    try:
        _old_rp.destroy()
        print("destroyed stale render product")
    except Exception:
        pass
writer.initialize(output_dir=OUTPUT, rgb=True)

# ---- camera: reuse if it exists (keeps your editor-tuned optics) ----
if not stage.GetPrimAtPath(CAM_PATH).IsValid():
    cam = UsdGeom.Camera.Define(stage, CAM_PATH)
    cam.GetFocalLengthAttr().Set(10.0)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
cam_prim = stage.GetPrimAtPath(CAM_PATH)

rp = rep.create.render_product(CAM_PATH, RES)
_rp = rp                       # remembered so the next run can destroy it
writer.attach([rp])


def look_at_matrix(eye, target, up=np.array([0.0, 0.0, 1.0])):
    eye = np.asarray(eye, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd                                   # USD camera looks down -Z
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye   # columns = axes
    return m


def set_cam_pose(eye):
    m = look_at_matrix(eye, TARGET)
    xfo = UsdGeom.Xformable(cam_prim)
    xfo.ClearXformOpOrder()
    # USD Matrix4d is ROW-MAJOR with translate in the last row -> transpose
    xfo.AddTransformOp().Set(Gf.Matrix4d(*m.T.flatten().tolist()))


positions = [
    (
        RADIUS * math.cos(2 * math.pi * i / NUM_VIEWS),
        RADIUS * math.sin(2 * math.pi * i / NUM_VIEWS),
        HEIGHT,
    )
    for i in range(NUM_VIEWS)
]

# ---- cancel a still-running previous coroutine (Script Editor keeps globals) ----
try:
    if _render_task is not None and not _render_task.done():
        _render_task.cancel()
        print("cancelled previous render task")
except NameError:
    pass


async def render_all():
    tl = omni.timeline.get_timeline_interface()
    tl.stop()                                     # start from a clean state
    tl.play()                                     # capture requires a running timeline
    for i, eye in enumerate(positions):
        set_cam_pose(eye)
        await rep.orchestrator.step_async()
    for _ in range(5):
        await rep.orchestrator.step_async()
    try:
        await rep.orchestrator.wait_until_complete_async()   # flush PNGs to disk
    except Exception:
        pass
    tl.stop()                                     # release the GPU, no manual stop needed
    n = len([f for f in os.listdir(OUTPUT) if f.endswith(".png")])
    print(f"DONE - timeline stopped, wrote {n} pngs (expect 41). Now run 05_export_poses.py")


_render_task = omni.kit.async_engine.run_coroutine(render_all())
print("rendering started (async)")
