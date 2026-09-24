"""T3d  Quick mesh viewer / inspector.

Usage:  python 16_view_mesh.py <mesh.obj|ply> [--shot out.png]

Prints the bbox so you can sanity-check the scale without a window, then opens
an interactive window. With --shot it renders offscreen instead (useful when
the GL window fails on this machine).
"""

import sys
import numpy as np
import open3d as o3d

path = sys.argv[1]
mesh = o3d.io.read_triangle_mesh(path)
if len(mesh.vertices) == 0:
    raise SystemExit("failed to load mesh: " + path)
mesh.compute_vertex_normals()

v = np.asarray(mesh.vertices)
size = v.max(0) - v.min(0)
print(f"verts {len(mesh.vertices)}  tris {len(mesh.triangles)}")
print(f"bbox  min {np.round(v.min(0), 4)}  max {np.round(v.max(0), 4)}")
print(f"size  x={size[0]:.4f} y={size[1]:.4f} z={size[2]:.4f} m")

if "--shot" in sys.argv:
    out = sys.argv[sys.argv.index("--shot") + 1]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        for ax, (a, b, t) in zip(
            axs, [(0, 1, "top (x-y)"), (0, 2, "front (x-z)"), (1, 2, "side (y-z)")]
        ):
            ax.scatter(v[:, a], v[:, b], s=0.3, c="steelblue")
            ax.set_title(t)
            ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print("saved orthographic views:", out)
    except Exception as e:
        print("offscreen render failed:", e)
else:
    o3d.visualization.draw_geometries([mesh], window_name=path)
