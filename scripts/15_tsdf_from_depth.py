"""T3c  TSDF fusion from the TRUE depth maps (ground-truth quality mesh).

Far better than poisson for a box: poisson inflated it by ~10 mm per side and
rounded the corners, TSDF keeps flat faces flat and edges sharp.

Everything here is in REAL metres already:
  * poses      from transforms.json (Isaac Sim ground truth)
  * intrinsics from transforms.json (real focal length)
  * depth      distance_to_image_plane, in metres

open3d's TSDF uses the OpenCV convention (y down, z forward), which is exactly
what our c2w matrices use, so extrinsic = inv(c2w).
"""

import os
import json
import numpy as np
import open3d as o3d
from PIL import Image

DATA = r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb"
OUT_DIR = r"C:\Users\theozhu\Desktop\拓展\3DGS\exports\ycb"
VOXEL = 0.002          # 2 mm
SDF_TRUNC = 0.012
DEPTH_TRUNC = 0.8      # ignore the far background

# object crop (real-world metres, known from the YCB asset)
HALF_X, HALF_Y = 0.090, 0.115
Z_LOW, Z_HIGH = 0.030, 0.080

# transforms.json now stores OPENGL c2w (07_build_init.py dropped the
# diag(1,-1,-1) flip on 2026-09-22 for nerfstudio's sake). Open3D's TSDF wants
# OPENCV, so flip back -- without this every surface is mirrored behind the
# camera (z ~0.23-0.58) and the object crop returns nothing.
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

tf = json.load(open(os.path.join(DATA, "transforms.json")))
W, H, fx, fy = tf["w"], tf["h"], tf["fl_x"], tf["fl_y"]
cx, cy = tf["cx"], tf["cy"]
intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
print(f"intrinsics {W}x{H} fx={fx:.1f} fy={fy:.1f}")

vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL,
    sdf_trunc=SDF_TRUNC,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

n = 0
for fr in tf["frames"]:
    img_p = os.path.join(DATA, fr["file_path"])
    dep_p = os.path.join(DATA, "depth", os.path.basename(fr["file_path"]).replace(".png", ".npy"))
    if not (os.path.exists(img_p) and os.path.exists(dep_p)):
        continue
    rgb = np.asarray(Image.open(img_p).convert("RGB"))
    depth = np.load(dep_p).astype(np.float32)
    color = o3d.geometry.Image(rgb)
    dimg = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, dimg, depth_trunc=DEPTH_TRUNC, depth_scale=1.0,
        convert_rgb_to_intensity=False,
    )
    c2w = np.array(fr["transform_matrix"], dtype=np.float64)
    vol.integrate(rgbd, intr, np.linalg.inv(c2w @ FLIP))   # -> OpenCV
    n += 1
print(f"integrated {n} frames")

mesh = vol.extract_triangle_mesh()
mesh.compute_vertex_normals()
print(f"raw mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

# crop to the object
v = np.asarray(mesh.vertices)
m = (
    (np.abs(v[:, 0]) < HALF_X)
    & (np.abs(v[:, 1]) < HALF_Y)
    & (v[:, 2] > Z_LOW)
    & (v[:, 2] < Z_HIGH)
)
tri = np.asarray(mesh.triangles)
keep = m[tri].all(axis=1)          # per-TRIANGLE mask
mesh.remove_triangles_by_mask(~keep)
mesh.remove_unreferenced_vertices()
print(f"after object crop: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

if len(mesh.triangles):
    clusters, n_tris, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    n_tris = np.asarray(n_tris)
    largest = int(np.argmax(n_tris))
    mask = clusters != largest
    if mask.any():
        mesh.remove_triangles_by_mask(mask)
        mesh.remove_unreferenced_vertices()
    print(f"largest component: {int(n_tris[largest])} tris")

v = np.asarray(mesh.vertices)
if len(v):
    size = v.max(0) - v.min(0)
    print(f"\nGT mesh bbox (m): x={size[0]:.3f} y={size[1]:.3f} z={size[2]:.3f}")
    print("YCB official    : x=0.164 y=0.213 z=0.072")
    err = max(abs(a - b) for a, b in zip(sorted((0.164, 0.213)), sorted(size[:2])))
    print(f"x/y error: {err*1000:.1f} mm -> " + ("OK" if err < 0.02 else "CHECK"))

os.makedirs(OUT_DIR, exist_ok=True)
p = os.path.join(OUT_DIR, "gt_mesh_tsdf.obj")
o3d.io.write_triangle_mesh(p, mesh)
print("saved:", p)
