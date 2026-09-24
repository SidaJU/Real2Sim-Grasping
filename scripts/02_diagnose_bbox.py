"""T1.1b  Diagnose an object's world bounding box (size in meters + position).

Use when the object is not visible in the viewport, or to check whether it is
sunk into the table.

Run in Isaac Sim > Window > Script Editor (after loading the object).
"""

import omni.usd
from pxr import UsdGeom, Usd

stage = omni.usd.get_context().get_stage()
obj = stage.GetPrimAtPath("/World/Object")

bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
)
rng = bbox_cache.ComputeWorldBound(obj).ComputeAlignedRange()

mpu = stage.GetMetadata("metersPerUnit") or 1.0
print("min  (m):", [v * mpu for v in rng.GetMin()])
print("max  (m):", [v * mpu for v in rng.GetMax()])
print("size (m):", [v * mpu for v in rng.GetSize()])
