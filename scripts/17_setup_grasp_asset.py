"""T4-1  Import the GT mesh into Isaac Sim as a graspable physical asset.

Does four things:
  1. convert  gt_mesh.obj -> gt_mesh.usd   (kit asset converter)
  2. reference it into the stage at /World/GraspObject
  3. add a rigid body + CONVEX HULL collider
     (a box is convex, so convexHull is an exact fit - no decomposition needed)
  4. print the resulting bbox so you can confirm it is still 0.164 x 0.213 m

Coordinates: the mesh already carries REAL-WORLD coordinates
(x ~ [-0.082, 0.082], y ~ [-0.106, 0.106], z ~ [0.030, 0.072]),
so it is placed at the origin with no transform.
"""

import omni.usd
import omni.kit.asset_converter
import omni.kit.async_engine
from omni.kit.asset_converter import AssetConverterContext
from pxr import Usd, UsdPhysics, UsdGeom, Gf

OBJ = r"C:\isaacsim_data\mesh\gt_mesh.obj"
USD = r"C:\isaacsim_data\mesh\gt_mesh.usd"
PRIM = "/World/GraspObject"
MASS_KG = 0.4          # a cracker box


async def main():
    # ---- 1. obj -> usd ----
    ctx = AssetConverterContext()
    ctx.single_mesh = True
    ctx.smooth_normals = True
    ctx.ignore_materials = False          # keep the texture
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        OBJ, USD, None, ctx
    )
    ok = await task.wait_until_finished()
    print(f"convert ok={ok} status={task.get_status()} err={task.get_error_message()}")
    if not ok:
        return

    stage = omni.usd.get_context().get_stage()

    # ---- 2. put it in the stage ----
    if stage.GetPrimAtPath(PRIM).IsValid():
        stage.RemovePrim(PRIM)
    ref = stage.OverridePrim(PRIM)
    ref.GetReferences().AddReference(USD)
    # offset so it does not collide with the original YCB box already at the origin
    from pxr import Gf as _Gf
    UsdGeom.Xformable(ref).ClearXformOpOrder()
    UsdGeom.Xformable(ref).AddTranslateOp().Set(_Gf.Vec3d(0.30, 0.0, 0.0))
    print("referenced", PRIM, "->", USD, "at x=+0.30")

    # ---- 3. physics: rigid body + convex hull (fully diagnosed) ----
    try:
        root = stage.GetPrimAtPath(PRIM)
        if not root.IsValid():
            print("ERROR: prim is invalid:", PRIM)
            return

        UsdPhysics.RigidBodyAPI.Apply(root)
        print("rigid body applied")

        meshes = [p for p in Usd.PrimRange(root) if p.GetTypeName() == "Mesh"]
        print("mesh prims found:", len(meshes))
        if not meshes:
            print("--- prim tree (for debugging) ---")
            for p in Usd.PrimRange(root):
                print("   ", p.GetPath(), "|", p.GetTypeName())
            print("--- no Mesh found, stopping here ---")
            return

        for m in meshes:
            UsdPhysics.CollisionAPI.Apply(m)
            col = UsdPhysics.MeshCollisionAPI.Apply(m)
            col.GetApproximationAttr().Set("convexHull")
            print("  collider:", m.GetPath())

        mass = UsdPhysics.MassAPI.Apply(root)
        mass.GetMassAttr().Set(MASS_KG)
        print(f"mass set: {MASS_KG} kg")
    except Exception:
        import traceback
        traceback.print_exc()
        return

    # ---- 4. verify the size ----
    bbox = UsdGeom.Boundable(ref).ComputeWorldBound(0.0, "default")
    rng = bbox.ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    size = Gf.Vec3d(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
    print(f"\nbbox min {mn}  max {mx}")
    print(f"size  x={size[0]:.4f} y={size[1]:.4f} z={size[2]:.4f} m")
    print("expect x=0.1642 y=0.2130 z=0.0422")
    print("\nnext: play the timeline - the box should rest on the table")


omni.kit.async_engine.run_coroutine(main())
print("import started (async)")
