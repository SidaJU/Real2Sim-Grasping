"""T1.2a  Build the capture scene: a table plus the 003_cracker_box object.

Run in Isaac Sim > Window > Script Editor on a NEW stage (File > New).
Verify visually: object sits at the table center at a hand-graspable scale.

Note: YCB models are centered on their geometry origin, so the object is half
sunk into the table until lifted (see 04_render_multiview.py, step 1).
"""

import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.experimental.utils.stage import add_reference_to_stage

stage = omni.usd.get_context().get_stage()
u = 1.0 / (stage.GetMetadata("metersPerUnit") or 1.0)


def box(path, size_m, pos_m, dynamic=False):
    """Create a collision box. size/pos are in METERS."""
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
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())


# Table: 2m x 2m slab, top surface at z = 0
box("/World/Table", (2, 2, 0.05), (0, 0, -0.025))

# Load the cracker box
root = get_assets_root_path()
add_reference_to_stage(
    usd_path=root + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    path="/World/Object",
)
print("loaded")
