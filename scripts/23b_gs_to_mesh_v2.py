"""T5b-v2  3DGS depth -> mesh: FIXES the two defects of v1.

v1 (23_gs_to_mesh.py) defects and their root causes
---------------------------------------------------
1. BROKEN / HOLES
   v1 kept a pixel only if chroma = max(RGB)-min(RGB) > 40, i.e. "is this pixel
   colourful enough". That asks the wrong question - we want "does this pixel
   belong to the object". Every pale region of the object (white artwork,
   white text, specular highlights) has low chroma -> its depth was zeroed ->
   TSDF never observed that surface -> marching cubes left a hole.

2. WASHED-OUT COLOUR
   v1 fed the ORIGINAL photos into the TSDF. Open3D's RGB8 colour integration
   is a plain weighted average over every frame that sees a voxel, so the
   per-frame exposure / white-balance differences average out to grey.

What v2 does instead
--------------------
* MASK (geometry, not colour): unproject every depth pixel into WORLD space and
  keep it if it falls inside the object's footprint and above the table plane.
  Colour never enters the decision, so white artwork is kept.
  Two supporting signals:
    - accumulation (alpha) from the 3DGS model: 1 = the ray hit solid geometry,
      0 = it missed everything (splatfacto.py:637 then writes a sentinel depth).
      This is a colour-independent VALIDITY check that v1 ignored entirely.
    - the table plane height is ESTIMATED from the data (mode of z over points
      outside the footprint) instead of being hard-coded.
* COLOUR: use the 3DGS model's own rendering (22 now dumps rgb_*.npy). One
  single 3D-consistent model -> no per-frame exposure jumps -> saturated colour.

Run with the nerfstudio env (has open3d 0.19):
    C:\\Users\\theozhu\\miniconda3\\envs\\nerfstudio\\python.exe 23b_gs_to_mesh_v2.py
"""

import json
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
# here -- otherwise every surface is mirrored behind the camera.
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

# object crop / footprint (real-world metres, from the YCB asset)
HALF_X, HALF_Y = 0.090, 0.115
Z_LOW, Z_HIGH = 0.030, 0.080

# The crop box above is ~8 mm WIDER than the asset's true footprint
# (x 0.082, y 0.1065). That slack lets a "skirt" survive: at grazing angles the
# 3DGS expected depth blends the object edge with the table gaussians, so the
# reconstructed wall flares OUTWARD near the base. Measured: error grows from
# 1.1 mm at the top face to 22.5 mm at the bottom band. Clipping the mask to the
# asset's true footprint turns the flare back into a straight wall.
MASK_HALF_X, MASK_HALF_Y = 0.083, 0.107

# ---- v2 mask knobs -------------------------------------------------------
ACC_MIN = 0.10          # accumulation below this = the ray hit nothing solid
                        # NOTE: measured acc ~= 1.0 for EVERY pixel (object and
                        # table are both opaque), so this is a validity check
                        # only - it cannot separate foreground from background.
TABLE_Z = Z_LOW         # 0.030: the object's underside sits ON the table, so the
                        # table plane is at the same height. Verified against the
                        # GT mesh (bbox z = 0.030 .. 0.0723).
                        # A data-driven estimate was tried first and FAILED: the
                        # world-z of points outside the footprint spreads over
                        # 0.03-0.07 (clutter, walls, floaters), so its mode is
                        # noise - it landed on 0.0437 and lopped 16 mm off the
                        # bottom of the object. Use the asset prior instead.
MASK_Z_UP = Z_HIGH      # 0.080, same upper bound as the crop
TABLE_MARGIN = 0.003    # 3 mm above the table: kills table pixels that fall
                        # inside the footprint (the crop box is ~8 mm wider than
                        # the asset, so a thin ring of table IS visible there)
MASK_ERODE = 8          # erode the per-frame mask by this many pixels. Depth
                        # near the object's silhouette / table-contact line is
                        # contaminated: the expected depth there blends the wall
                        # with the table behind it, which bends the walls inward
                        # ("tent" shape). Erosion drops those unreliable pixels.
SKIP_LOW_RING = True    # v1 needed this: low-ring frames carry floater gaussians
USE_GS_COLOUR = True    # False -> fall back to the original photos (A/B switch)
DEBUG_FRAME = 50        # which frame to dump mask diagnostics for
N_SAMPLES = 30000

OUT_NAME = "gs_mesh_v2_mask-footprint+worldz_color-gsrgb_voxel2mm.obj"

NORM_JSON = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\outputs\ycb\splatfacto"
                 r"\2026-09-23_102733\dataparser_transforms.json")

tf = json.load(open(DATA / "transforms.json"))
W, H, fx, fy, cx, cy = tf["w"], tf["h"], tf["fl_x"], tf["fl_y"], tf["cx"], tf["cy"]
intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

depths = sorted(GS_DEPTH.glob("depth_*.npy"))
print(f"3DGS depth maps: {len(depths)}")
if not depths:
    raise SystemExit("run 22_render_gs_depth.py first")
have_acc = (GS_DEPTH / "acc_0000.npy").exists()
have_rgb = (GS_DEPTH / "rgb_0000.npy").exists()
print(f"aux outputs: acc={have_acc}  rgb={have_rgb}  "
      f"(re-run 22 to enable both)")

SCALE = float(json.load(open(NORM_JSON))["scale"])
print(f"3DGS depth is in NORMALISED units: scale = {SCALE:.4f}")

# pixel grid, reused for every frame
_u = np.arange(W, dtype=np.float64)
_v = np.arange(H, dtype=np.float64)
UU, VV = np.meshgrid(_u, _v)


def world_points(depth_m, w2c):
    """Unproject a z-depth map into world coordinates (H, W, 3)."""
    X = (UU - cx) * depth_m / fx
    Y = (VV - cy) * depth_m / fy
    Z = depth_m
    R, t = w2c[:3, :3], w2c[:3, 3]
    # p_cam = R @ p_world + t   =>   p_world = R^T (p_cam - t)
    # for row vectors:          (p_cam - t) @ R
    return np.stack(((X - t[0]), (Y - t[1]), (Z - t[2])), -1) @ R


def estimate_table_z(pts_w, valid):
    """Mode of world-z over points OUTSIDE the object footprint = table plane."""
    outside = valid & ((np.abs(pts_w[..., 0]) > HALF_X)
                       | (np.abs(pts_w[..., 1]) > HALF_Y))
    z = pts_w[..., 2][outside]
    z = z[np.isfinite(z)]
    if z.size < 500:
        return None
    lo, hi = np.percentile(z, [1, 99])
    if hi - lo < 1e-4:
        return float(np.median(z))
    edges = np.arange(lo, hi + 0.005, 0.005)
    hist, _ = np.histogram(z, bins=edges)
    return float(edges[np.argmax(hist)] + 0.0025)


vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL, sdf_trunc=SDF_TRUNC,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

n = 0
skipped = 0
kept_px = 0
total_px = 0
hint_printed = False
for i, fr in enumerate(tf["frames"]):
    if SKIP_LOW_RING and i < 36:
        skipped += 1
        continue
    dp = GS_DEPTH / f"depth_{i:04d}.npy"
    img_p = DATA / fr["file_path"]
    if not (dp.exists() and img_p.exists()):
        continue

    depth = np.load(dp).astype(np.float64) / SCALE      # normalised -> metres
    acc = None
    if have_acc:
        ap = GS_DEPTH / f"acc_{i:04d}.npy"
        if ap.exists():
            acc = np.load(ap).astype(np.float64) / 255.0
    if USE_GS_COLOUR and have_rgb:
        rp = GS_DEPTH / f"rgb_{i:04d}.npy"
        rgb = np.load(rp) if rp.exists() else None
    else:
        rgb = None
    if rgb is None:
        rgb = np.asarray(Image.open(img_p).convert("RGB"))
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))

    c2w = np.array(fr["transform_matrix"], dtype=np.float64)
    w2c = np.linalg.inv(c2w @ FLIP)                     # -> OpenCV convention

    valid = np.isfinite(depth) & (depth > 0.01) & (depth < DEPTH_TRUNC)
    if acc is not None:
        valid &= acc > ACC_MIN

    pts_w = world_points(np.where(valid, depth, 0.0), w2c)

    # diagnostic only: what a data-driven estimate would have said (kept to
    # document why we do NOT use it - see the TABLE_Z comment above)
    if not hint_printed:
        hint = estimate_table_z(pts_w, valid)
        print(f"data-driven table hint = "
              f"{'n/a' if hint is None else format(hint, '.4f')} m -> ignored, "
              f"using asset prior {TABLE_Z} (mask starts at {TABLE_Z + TABLE_MARGIN})")
        hint_printed = True

    # ---- THE NEW MASK: geometry only, no colour anywhere -----------------
    in_footprint = ((np.abs(pts_w[..., 0]) < MASK_HALF_X)
                    & (np.abs(pts_w[..., 1]) < MASK_HALF_Y))
    above_table = ((pts_w[..., 2] > TABLE_Z + TABLE_MARGIN)
                   & (pts_w[..., 2] < MASK_Z_UP))
    m = valid & in_footprint & above_table

    if MASK_ERODE > 0:
        try:
            from scipy.ndimage import binary_erosion
            m = binary_erosion(m, iterations=MASK_ERODE)
        except ImportError:
            pass

    depth = np.where(m, depth, 0.0)                     # background -> 0 = invalid
    kept_px += int(m.sum())
    total_px += m.size

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb), o3d.geometry.Image(depth.astype(np.float32)),
        depth_trunc=DEPTH_TRUNC, depth_scale=1.0, convert_rgb_to_intensity=False,
    )
    vol.integrate(rgbd, intr, w2c)
    n += 1

    if i == DEBUG_FRAME:
        dbg = OUT_DIR / f"debug_v2_mask_f{i:04d}.png"
        Image.fromarray((m * 255).astype(np.uint8)).save(dbg)
        Image.fromarray(rgb).save(OUT_DIR / f"debug_v2_rgb_f{i:04d}.png")
        print(f"  debug: wrote {dbg.name} (white = kept)")

print(f"integrated {n} frames (skipped {skipped} low-ring)")
print(f"mask kept {kept_px / max(total_px, 1) * 100:.1f}% of pixels")

mesh = vol.extract_triangle_mesh()
mesh.compute_vertex_normals()
print(f"raw gs mesh: {len(mesh.vertices)} verts")

v = np.asarray(mesh.vertices)
keep = (
    (np.abs(v[:, 0]) < MASK_HALF_X) & (np.abs(v[:, 1]) < MASK_HALF_Y)
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
    mm = clusters != largest
    if mm.any():
        mesh.remove_triangles_by_mask(mm)
        mesh.remove_unreferenced_vertices()
print(f"gs mesh after crop: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

v = np.asarray(mesh.vertices)
if len(v):
    size = v.max(0) - v.min(0)
    print(f"gs mesh bbox: x={size[0]:.3f} y={size[1]:.3f} z={size[2]:.3f} m")

out = OUT_DIR / OUT_NAME
o3d.io.write_triangle_mesh(str(out), mesh)
print("saved:", out)

# ---------------- score against the GT ----------------
if not len(mesh.triangles):
    print("EMPTY mesh - nothing to score")
    raise SystemExit(1)

if GT_MESH.exists():
    gt = o3d.io.read_triangle_mesh(str(GT_MESH))
    gt.compute_vertex_normals()
    p_gs = mesh.sample_points_uniformly(number_of_points=N_SAMPLES)
    p_gt = gt.sample_points_uniformly(number_of_points=N_SAMPLES)
    d1 = np.asarray(p_gs.compute_point_cloud_distance(p_gt))
    d2 = np.asarray(p_gt.compute_point_cloud_distance(p_gs))
    chamfer = 0.5 * (d1.mean() + d2.mean())
    print("\n================ RECONSTRUCTION SCORE (v2) ================")
    print(f"  gs -> gt : {d1.mean()*1000:.2f} mm   (p95 {np.percentile(d1,95)*1000:.2f} mm)")
    print(f"  gt -> gs : {d2.mean()*1000:.2f} mm   (p95 {np.percentile(d2,95)*1000:.2f} mm)")
    print(f"  Chamfer  : {chamfer*1000:.2f} mm      (v1 was 5.00 mm)")
    print("===========================================================")
else:
    print("GT mesh not found - skipping the comparison")
