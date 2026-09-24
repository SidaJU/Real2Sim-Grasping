"""T5b  3DGS depth -> mesh, then score it against the GT mesh.

This produces the number that makes the whole project credible:
    how many millimetres apart is the RGB-only reconstruction from the
    ground-truth asset, and is it good enough to grasp?

Both meshes live in the SAME real-world frame (same poses, same intrinsics,
same metres), so they can be compared directly.
"""

import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

DATA = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb")
GS_DEPTH = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\exports\ycb\gs_depth")
OUT_DIR = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\exports\ycb")
GT_MESH = Path(r"C:\isaacsim_data\mesh\gt_mesh.obj")   # ASCII path: open3d
                                                       # cannot read the UTF-8 one

VOXEL, SDF_TRUNC, DEPTH_TRUNC = 0.002, 0.012, 0.8

# transforms.json stores OPENGL c2w (07_build_init.py stopped applying the
# diag(1,-1,-1) flip on 2026-09-22 because nerfstudio wants OpenGL). Open3D's
# TSDF integrator however expects the OPENCV convention, so we must flip back
# here -- otherwise every surface is mirrored behind the camera and the object
# crop ends up empty (verified: z 0.227-0.575 without, 0.025-0.072 with).
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
HALF_X, HALF_Y = 0.090, 0.115
Z_LOW, Z_HIGH = 0.030, 0.080
N_SAMPLES = 30000

tf = json.load(open(DATA / "transforms.json"))
W, H, fx, fy, cx, cy = tf["w"], tf["h"], tf["fl_x"], tf["fl_y"], tf["cx"], tf["cy"]
intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

depths = sorted(GS_DEPTH.glob("*.npy"))
print(f"3DGS depth maps: {len(depths)}")
if not depths:
    raise SystemExit("run 22_render_gs_depth.py first")

# ------------------------------------------------------------------
# The model was trained in nerfstudio's NORMALISED space (auto-scale-poses),
# so its depths are in normalised units, not metres.
#
# Two independent ways to recover the scale:
#   (a) EXACT - read it from dataparser_transforms.json (uses NO ground truth)
#   (b) CHECK - ratio against the true depth maps (sanity check only)
# If (a) and (b) agree, the depths AND the normalisation are both consistent.
# ------------------------------------------------------------------
NORM_JSON = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\outputs\ycb\splatfacto"
                 r"\2026-09-23_102733\dataparser_transforms.json")
SCALE = float(json.load(open(NORM_JSON))["scale"])
print(f"3DGS depth is in NORMALISED units: scale = {SCALE:.4f} "
      f"(from dataparser_transforms.json, no GT used)")

GT_DEPTH = DATA / "depth"
ratios = []
for i in range(0, len(depths), max(1, len(depths) // 20)):
    gt_p = GT_DEPTH / f"rgb_{i:04d}.npy"      # 07 wrote GT depth with the image name
    gs_p = GS_DEPTH / f"depth_{i:04d}.npy"
    if not (gt_p.exists() and gs_p.exists()):
        continue
    gt = np.load(gt_p).astype(np.float32)
    gs = np.load(gs_p).astype(np.float32)
    m = np.isfinite(gt) & np.isfinite(gs) & (gt > 0.05) & (gs > 0.05)
    if m.sum() > 1000:
        ratios.append(float(np.median(gs[m]) / np.median(gt[m])))
if ratios:
    chk = float(np.median(ratios))
    print(f"cross-check vs true depth: scale = {chk:.4f} "
          f"(agreement {abs(chk - SCALE) / SCALE * 100:.2f}%)")

vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL, sdf_trunc=SDF_TRUNC,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

# Low ring (frames 0..35) renders the object at table depth: near-camera
# floater gaussians dominate the expected depth there (this run was trained
# WITHOUT a mask). Measured against the true depth, low-ring object depth comes
# out 0.05 m instead of 0.15 m, while mid/top rings agree to 0-7%.
SKIP_LOW_RING = True          # set False to fuse all 99 frames

n = 0
masked_px = 0
total_px = 0
skipped = 0
for i, fr in enumerate(tf["frames"]):
    if SKIP_LOW_RING and i < 36:
        skipped += 1
        continue
    dp = GS_DEPTH / f"depth_{i:04d}.npy"
    img_p = DATA / fr["file_path"]
    if not (dp.exists() and img_p.exists()):
        continue
    depth = np.load(dp).astype(np.float32) / SCALE    # normalised -> metres
    rgb = np.asarray(Image.open(img_p).convert("RGB"))

    # 3DGS keeps background gaussians (they were never told not to), so its
    # depth is valid-looking fog over the white table. Fusing that produces a
    # multi-million-vertex blob that buries the object. Kill it with an RGB
    # chroma mask -- the box is saturated red/yellow, the table is grey/white.
    # No ground truth involved: a plain-colour cell is standard on a real line.
    chroma = rgb.max(2).astype(np.int16) - rgb.min(2).astype(np.int16)
    m = chroma > 40
    try:
        from scipy.ndimage import binary_fill_holes
        m = binary_fill_holes(m)
    except ImportError:
        pass
    masked_px += int(m.sum())
    total_px += m.size
    depth = depth * m                                  # background -> 0 = invalid
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
        depth_trunc=DEPTH_TRUNC, depth_scale=1.0, convert_rgb_to_intensity=False,
    )
    c2w = np.array(fr["transform_matrix"], dtype=np.float64)
    vol.integrate(rgbd, intr, np.linalg.inv(c2w @ FLIP))   # -> OpenCV
    n += 1
print(f"integrated {n} frames (skipped {skipped} low-ring)")
print(f"mask kept {masked_px / max(total_px, 1) * 100:.1f}% of pixels "
      f"(object only, background depth zeroed)")

mesh = vol.extract_triangle_mesh()
mesh.compute_vertex_normals()
print(f"raw gs mesh: {len(mesh.vertices)} verts")

v = np.asarray(mesh.vertices)
keep = (
    (np.abs(v[:, 0]) < HALF_X) & (np.abs(v[:, 1]) < HALF_Y)
    & (v[:, 2] > Z_LOW) & (v[:, 2] < Z_HIGH)
)
tri = np.asarray(mesh.triangles)
mesh.remove_triangles_by_mask(~keep[tri].all(axis=1))
mesh.remove_unreferenced_vertices()

if len(mesh.triangles):
    clusters, n_tris, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    n_tris = np.asarray(n_tris)
    largest = int(np.argmax(n_tris))
    m = clusters != largest
    if m.any():
        mesh.remove_triangles_by_mask(m)
        mesh.remove_unreferenced_vertices()
print(f"gs mesh after crop: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

v = np.asarray(mesh.vertices)
if len(v):
    size = v.max(0) - v.min(0)
    print(f"gs mesh bbox: x={size[0]:.3f} y={size[1]:.3f} z={size[2]:.3f} m")

out = OUT_DIR / "gs_mesh.obj"
o3d.io.write_triangle_mesh(str(out), mesh)
print("saved:", out)

# ---------------- score against the GT ----------------
if not len(mesh.triangles):
    print("EMPTY mesh - nothing to score (check mask coverage / depth scale above)")
    raise SystemExit(1)

if GT_MESH.exists():
    gt = o3d.io.read_triangle_mesh(str(GT_MESH))
    gt.compute_vertex_normals()
    p_gs = mesh.sample_points_uniformly(number_of_points=N_SAMPLES)
    p_gt = gt.sample_points_uniformly(number_of_points=N_SAMPLES)
    d1 = np.asarray(p_gs.compute_point_cloud_distance(p_gt))
    d2 = np.asarray(p_gt.compute_point_cloud_distance(p_gs))
    chamfer = 0.5 * (d1.mean() + d2.mean())
    print("\n================ RECONSTRUCTION SCORE ================")
    print(f"  gs -> gt : {d1.mean()*1000:.2f} mm   (p95 {np.percentile(d1,95)*1000:.2f} mm)")
    print(f"  gt -> gs : {d2.mean()*1000:.2f} mm   (p95 {np.percentile(d2,95)*1000:.2f} mm)")
    print(f"  Chamfer  : {chamfer*1000:.2f} mm")
    print("=====================================================")
else:
    print("GT mesh not found - skipping the comparison")
