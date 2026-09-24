"""T3b  Build the GROUND-TRUTH mesh from the true-depth point cloud.

points3D.ply is in REAL-WORLD metres (back-projected from Isaac Sim's true
depth), so unlike the 3DGS mesh it needs no rescaling. We can also crop the
object precisely because we know where it is:

    table top  z = 0.0252 m
    box        x in [-0.082, +0.082], y in [-0.106, +0.106]   (0.164 x 0.213 m)
    box top    z = 0.072 m   (bottom is buried in the table)

This mesh serves two purposes:
  1. a usable asset for grasp planning (T4)
  2. the reference for scoring the 3DGS reconstruction (Chamfer distance)
"""

import os
import numpy as np
import open3d as o3d

SRC = r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb\points3D.ply"
OUT_DIR = r"C:\Users\theozhu\Desktop\拓展\3DGS\exports\ycb"
TABLE_Z = 0.0252
HALF_X, HALF_Y = 0.082, 0.1065      # half of 0.164 x 0.213 (with a little slack)
Z_LOW, Z_HIGH = TABLE_Z + 0.005, 0.080
POISSON_DEPTH = 10

pcd = o3d.io.read_point_cloud(SRC)
pts = np.asarray(pcd.points)
print(f"loaded {len(pts)} points (real metres)")

m = (
    (np.abs(pts[:, 0]) < HALF_X)
    & (np.abs(pts[:, 1]) < HALF_Y)
    & (pts[:, 2] > Z_LOW)
    & (pts[:, 2] < Z_HIGH)
)
print(f"object crop: {int(m.sum())} points")
if m.sum() < 500:
    raise SystemExit("too few object points - check the crop bounds")

obj = pcd.select_by_index(np.where(m)[0])
print(f"bbox before clean: min {np.round(obj.get_min_bound(),3)} "
      f"max {np.round(obj.get_max_bound(),3)}")

# statistical filter for leftover stragglers
obj, _ = obj.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.5)
print(f"after outlier removal: {len(obj.points)} points")

obj.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
obj.orient_normals_to_align_with_direction((0.0, 0.0, 1.0))

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    obj, depth=POISSON_DEPTH
)
mesh.compute_vertex_normals()
print(f"mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

# largest component
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
size = v.max(0) - v.min(0)
print(f"\nGT mesh bbox (m): x={size[0]:.3f} y={size[1]:.3f} z={size[2]:.3f}")
print(f"YCB official    : x=0.164 y=0.213 z=0.072")
err = max(abs(a - b) for a, b in zip(sorted((0.164, 0.213)), sorted(size[:2])))
print(f"x/y error: {err*1000:.1f} mm -> " + ("OK" if err < 0.02 else "CHECK"))

os.makedirs(OUT_DIR, exist_ok=True)
for name in ("gt_mesh.obj", "gt_mesh.ply"):
    p = os.path.join(OUT_DIR, name)
    o3d.io.write_triangle_mesh(p, mesh)
    print("saved:", p)
