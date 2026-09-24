"""T4-2  Make sure a Franka is in the scene, parked in front of the GT box.

Layout (world metres, Z-up):
    GT box   at (0.30, 0, ~0.05)   facing +Y from the robot
    Franka   at (0.30, -0.60, 0)   rotated +90 deg about Z so it faces +Y
Base-to-box distance = 0.60 m, well inside Franka's ~0.85 m reach.

Run me, then press PLAY: the arm must hold its pose (joint drives active)
instead of collapsing. If it slumps, the drive gains need setting - tell me.
"""

import omni.usd
import omni.timeline
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.experimental.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom, Gf, UsdPhysics

PRIM = "/World/Franka"
BOX = (0.30, 0.0)          # where the GT box sits

stage = omni.usd.get_context().get_stage()

if stage.GetPrimAtPath(PRIM).IsValid():
    print("Franka already in scene at", PRIM)
else:
    root = get_assets_root_path()
    usd = root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    print("loading:", usd)
    add_reference_to_stage(usd_path=usd, path=PRIM)

    xf = UsdGeom.Xformable(stage.GetPrimAtPath(PRIM))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(BOX[0], BOX[1] - 0.60, 0.0))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 90.0))   # face +Y
    print("Franka placed at (0.30, -0.60, 0) facing +Y")

# sanity: report what is now on the stage
print("\n--- scene check ---")
for p in stage.Traverse():
    if p.GetPath().pathElementCount == 2:
        print(f"  {p.GetPath()}  [{p.GetTypeName()}]")
print("\nnext: press PLAY - the arm should stand still with drives holding it")
