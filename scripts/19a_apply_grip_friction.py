"""T4-3b  Bind a high-friction PHYSICS material to the grasp surfaces.

The Isaac 5.x UI no longer exposes "Physics Material" under + Add -> Physics,
and the PhysicsScene "default material" did not take effect on this scene.
So we author it directly in USD, which is all the UI was doing anyway:

  material attributes  physics:staticFriction  /  physics:dynamicFriction
                       physxMaterial:frictionCombineMode
  binding              relationship  material:binding:physics  on the COLLIDER

combine mode = "max" matters: it makes the contact friction
    max(box friction, finger friction)
so binding the BOX alone is enough - the fingers (inside an instanced
reference, which refuses edits) can keep their default.

Run this ONCE in the Script Editor (timeline may be stopped), then re-run 19.
"""

import omni.usd
from pxr import Sdf, UsdShade, UsdPhysics, PhysxSchema

MAT_PATH = "/World/PhysicsMaterial_Grip"
STATIC_FRICTION = 2.0
DYNAMIC_FRICTION = 2.0
RESTITUTION = 0.0

# prims that should use it. The robot ones are best-effort: the Franka is an
# instanceable reference, so edits there may be rejected - that is fine, the
# "max" combine mode picks up the box's value anyway.
TARGETS = [
    "/World/GraspObject",
    "/World/Franka/panda_leftfinger/geometry/panda_leftfinger",
    "/World/Franka/panda_rightfinger/geometry/panda_rightfinger",
]

stage = omni.usd.get_context().get_stage()

# ---- 1) create / author the material prim -------------------------------
mat_prim = stage.GetPrimAtPath(MAT_PATH)
if not mat_prim.IsValid():
    mat_prim = UsdShade.Material.Define(stage, MAT_PATH).GetPrim()
    print("created", MAT_PATH)
UsdPhysics.MaterialAPI.Apply(mat_prim)
PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)

for name, value, typ in (
    ("physics:staticFriction", STATIC_FRICTION, Sdf.ValueTypeNames.Float),
    ("physics:dynamicFriction", DYNAMIC_FRICTION, Sdf.ValueTypeNames.Float),
    ("physics:restitution", RESTITUTION, Sdf.ValueTypeNames.Float),
):
    a = mat_prim.CreateAttribute(name, typ)
    a.Set(value)
    print(f"  {name} = {value}")

for name, value in (
    ("physxMaterial:frictionCombineMode", "max"),
    ("physxMaterial:restitutionCombineMode", "max"),
):
    try:
        a = mat_prim.CreateAttribute(name, Sdf.ValueTypeNames.Token)
        a.Set(value)
        print(f"  {name} = {value}")
    except Exception as e:
        print(f"  [skip] {name}: {e}")

# ---- 2) bind it to the colliders ----------------------------------------
def apply_binding(path):
    p = stage.GetPrimAtPath(path)
    if not p.IsValid():
        print(f"  [skip] {path} (not found)")
        return False
    try:
        UsdShade.MaterialBindingAPI(p).Bind(
            UsdShade.Material(mat_prim),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        rel = p.GetRelationship("material:binding:physics")
        print(f"  bound {path} -> {MAT_PATH} ({rel.GetTargets()})")
        return True
    except Exception as e:
        print(f"  [fail] {path}: {e}")
        return False

print("\nbinding:")
for t in TARGETS:
    apply_binding(t)

print("\ndone. Verify in the Property panel of the box: "
      "Material (PhysX) should now list PhysicsMaterial_Grip.")
