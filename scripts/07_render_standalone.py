"""Standalone orbit renderer: build scene -> render N views -> write poses.

Run from a terminal (NOT the Script Editor):

    C:\\isaacsim\\python.bat C:\\Users\\theozhu\\Desktop\\拓展\\Real2Sim-Grasping\\scripts\\07_render_standalone.py

Why standalone instead of the Script Editor:
The Script Editor runs inside a long-lived Kit process where Replicator keeps
GLOBAL state (graph nodes, writer bindings, live coroutines). That state
outlives your script and corrupts later runs - stale render products keep
writing images, camera prims resurrect after deletion, frame numbers drift.
A standalone process starts clean and exits clean, so every run is identical
and synchronous step() is allowed (it is forbidden inside Kit).

Everything is configured by the constants below - there is no GUI to tweak.
"""

from isaacsim import SimulationApp

CONFIG = {
    "headless": False,      # True = faster, no window
    "width": 1280,
    "height": 720,
}
kit = SimulationApp(CONFIG)

import math
import os
import json
import numpy as np
import omni.usd
import omni.replicator.core as rep
import omni.timeline
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.experimental.utils.stage import add_reference_to_stage
from pxr import UsdGeom, UsdPhysics, UsdLux, Gf

# ==================== TUNABLE PARAMETERS ====================
OUTPUT = "C:/isaacsim_data/ycb_v2"
NUM_VIEWS = 36              # set to 1 while dialing in lighting
RADIUS, HEIGHT = 0.25, 0.15  # orbit radius / camera height (meters)
TARGET = np.array([0.0, 0.0, 0.036])   # object center
RES = (640, 640)
FOCAL_MM = 24.0
DOME_INTENSITY = 1.0        # if frames are white -> 0.3 ; if too dark -> 3.0
OBJECT_USD = "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd"
CAM_PATH = "/World/OrbitCamera"
# =============================================================

os.makedirs(OUTPUT, exist_ok=True)
for f in os.listdir(OUTPUT):
    if f.endswith(".png"):
        os.remove(os.path.join(OUTPUT, f))

stage = omni.usd.get_context().get_stage()
u = 1.0 / (stage.GetMetadata("metersPerUnit") or 1.0)   # stage units per meter


def box(path, size_m, pos_m):
    """Collision slab/box; size and pos are in meters."""
    UsdGeom.Xform.Define(stage, path)
    mesh = UsdGeom.Cube.Define(stage, path + "/Mesh")
    mesh.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(mesh.GetPrim())
    xf.ClearXformOpOrder()
    op = xf.AddTransformOp()
    m = Gf.Matrix4d()
    m.SetTranslate(Gf.Vec3d(pos_m[0] * u, pos_m[1] * u, pos_m[2] * u))
    m.SetScale(Gf.Vec3d(size_m[0] * u, size_m[1] * u, size_m[2] * u))
    op.Set(m)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


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


# ---------- scene ----------
box("/World/Table", (2, 2, 0.05), (0, 0, -0.025))       # table top at z = 0

root = get_assets_root_path()
add_reference_to_stage(usd_path=root + OBJECT_USD, path="/World/Object")
obj_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Object"))
obj_xf.ClearXformOpOrder()
obj_xf.AddTranslateOp().Set((0, 0, 0.036))              # rest on the table

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.GetIntensityAttr().Set(DOME_INTENSITY)

# ---------- camera ----------
cam = UsdGeom.Camera.Define(stage, CAM_PATH)
cam.GetFocalLengthAttr().Set(FOCAL_MM)
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))   # default near=1m clips the object

rp = rep.create.render_product(CAM_PATH, RES)
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=OUTPUT, rgb=True)
writer.attach([rp])


def set_cam_pose(eye):
    m = look_at_matrix(eye, TARGET)
    xfo = UsdGeom.Xformable(cam.GetPrim())
    xfo.ClearXformOpOrder()
    # USD Matrix4d is ROW-MAJOR with translate in the last row -> transpose
    xfo.AddTransformOp().Set(Gf.Matrix4d(*m.T.flatten().tolist()))


# ---------- render (synchronous: legal in standalone) ----------
omni.timeline.get_timeline_interface().play()

positions = []
for i in range(NUM_VIEWS):
    th = 2 * math.pi * i / NUM_VIEWS
    eye = (RADIUS * math.cos(th), RADIUS * math.sin(th), HEIGHT)
    positions.append(eye)
    set_cam_pose(eye)
    rep.orchestrator.step()

for _ in range(5):                 # flush the writer queue
    rep.orchestrator.step()
rep.orchestrator.wait_until_complete()
omni.timeline.get_timeline_interface().stop()

pngs = sorted([f for f in os.listdir(OUTPUT) if f.endswith(".png")])
print("rendered:", len(pngs))

# ---------- write nerfstudio transforms.json ----------
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])   # USD(OpenGL) -> OpenCV(nerfstudio)
APERTURE_MM = 20.955
fl_px = FOCAL_MM / APERTURE_MM * RES[0]

frames = []
for i, eye in enumerate(positions[: len(pngs)]):
    c2w = look_at_matrix(eye, TARGET) @ FLIP
    frames.append(
        {"file_path": f"images/{pngs[i]}", "transform_matrix": c2w.tolist()}
    )

with open(os.path.join(OUTPUT, "transforms.json"), "w") as f:
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
print("transforms.json:", len(frames), "frames ->", OUTPUT)

kit.close()
