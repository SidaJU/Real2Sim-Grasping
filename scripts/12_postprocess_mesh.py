"""T3  gaussian splat -> cleaned, real-scale mesh.

Pipeline
--------
ns-export gaussian-splat  ->  splat.ply (gaussian centres, NORMALISED space)
  1. statistical outlier removal   (kills the floater gaussians)
  2. normal estimation             (poisson needs oriented normals)
  3. poisson reconstruction        (-> watertight surface)
  4. low-density crop              (poisson's thin sheet artefacts)
  5. largest connected component   (final floater sweep)
  6. RESCALE TO REAL METRES        (see below)
  7. verify against YCB size

Why step 6 matters
------------------
nerfstudio normalises the scene before training, so the mesh size is arbitrary.
We know the true geometry from the true-depth point cloud:
    table top z = 0.0252 m ;  box top z = 0.0720 m  ->  visible height 0.047 m
So:
    scale = 0.047 / (z_max - z_table)      (both measured on the mesh)
    shift = 0.0252 - z_table * scale       (put the table back where it belongs)
"""

import sys
import os
import glob
import numpy as np
import open3d as o3d

TRUE_TABLE_Z = 0.0252
TRUE_BOX_TOP_Z = 0.0720
TRUE_VISIBLE_H = TRUE_BOX_TOP_Z - TRUE_TABLE_Z   # 0.0468 m
YCB_SIZE = (0.164, 0.213, 0.072)
POISSON_DEPTH = 11
DENSITY_QUANTILE = 0.02

export_dir = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/theozhu/Desktop/拓展/3DGS/exports/ycb"

# ---- locate the gaussian splat ----
cands = glob.glob(os.path.join(export_dir, "*.ply"))
if not cands:
    raise SystemExit(f"no .ply found in {export_dir} - run export_mesh.bat first")
src = max(cands, key=os.path.getmtime)
print("input:", src, f"({os.path.getsize(src)/1e6:.1f} MB)")

pcd = o3d.io.read_point_cloud(src)
print(f"points: {len(pcd.points)}")
if len(pcd.points) == 0:
    raise SystemExit("empty point cloud")

# ---- 1. drop statistical outliers (floaters are isolated in space) ----
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"after outlier removal: {len(pcd.points)}")

# ---- 2. normals ----
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_to_align_with_direction((0.0, 0.0, 1.0))

# ---- 3. poisson ----
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=POISSON_DEPTH
)
print(f"poisson mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

# ---- 4. crop low-density surface (poisson sheets in empty space) ----
dens = np.asarray(densities)
if dens.size:
    cut = np.quantile(dens, DENSITY_QUANTILE)
    mesh.remove_vertices_by_mask(dens < cut)
    print(f"after density crop: {len(mesh.vertices)} verts")

# ---- 5. largest connected component ----
clusters, n_tris, _ = mesh.cluster_connected_triangles()
clusters = np.asarray(clusters)
n_tris = np.asarray(n_tris)
largest = int(np.argmax(n_tris))
mask = clusters != largest
if mask.any():
    mesh.remove_triangles_by_mask(mask)
    mesh.remove_unreferenced_vertices()
print(f"largest component: {int(n_tris[largest])} tris "
      f"(dropped {int((~mask).sum())} tris)")

mesh.compute_vertex_normals()

# ---- 6. rescale to real-world metres ----
v = np.asarray(mesh.vertices)
z = v[:, 2]
hist, edges = np.histogram(z, bins=400)
k = int(np.argmax(hist))
z_table = 0.5 * (edges[k] + edges[k + 1])
z_top = float(z.max())
h_norm = z_top - z_table
if h_norm <= 0:
    raise SystemExit("cannot determine scale: degenerate mesh")
scale = TRUE_VISIBLE_H / h_norm
print(f"\nnormalised: table z={z_table:.4f} top z={z_top:.4f} height={h_norm:.4f}")
print(f"scale: {scale:.4f} mesh-units -> metres")

mesh.scale(scale, center=(0.0, 0.0, 0.0))
shift = TRUE_TABLE_Z - (z_table * scale)
mesh.translate((0.0, 0.0, shift))
print(f"shift z: {shift:+.4f} m -> table at z={TRUE_TABLE_Z}")

# ---- 7. verify ----
v = np.asarray(mesh.vertices)
obj = v[v[:, 2] > TRUE_TABLE_Z + 0.015]
if len(obj) > 100:
    size = obj.max(0) - obj.min(0)
    print(f"\nbox bbox (m): x={size[0]:.3f} y={size[1]:.3f} "
          f"height above table={size[2]:.3f}")
    print(f"YCB official: x={YCB_SIZE[0]:.3f} y={YCB_SIZE[1]:.3f} "
          f"thickness={YCB_SIZE[2]:.3f}")
    err = max(abs(a - b) for a, b in zip(sorted(YCB_SIZE[:2]), sorted(size[:2])))
    print(f"x/y error: {err*1000:.1f} mm -> " + ("OK" if err < 0.02 else "CHECK"))
else:
    print("WARNING: no geometry above the table")

out_obj = os.path.join(export_dir, "ycb_mesh_real_scale.obj")
o3d.io.write_triangle_mesh(out_obj, mesh)
print("\nsaved:", out_obj)
print("import this into Isaac Sim - it is in real metres, aligned to the USD scene")
