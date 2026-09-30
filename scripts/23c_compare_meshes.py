"""A/B compare two reconstructed meshes (v1 chroma-mask vs v2 geometry-mask).

Quantifies the two defects we set out to fix:
  * holes        -> number of BOUNDARY edges (edges used by only one triangle).
                    A closed surface has 0. Every hole adds a rim of them.
  * washed-out   -> mean chroma of the vertex colours,
                    chroma = max(RGB) - min(RGB), 0..255. Higher = more saturated.

Run:  C:\\Users\\theozhu\\miniconda3\\envs\\nerfstudio\\python.exe 23c_compare_meshes.py
"""

from pathlib import Path

import numpy as np
import open3d as o3d

# NOTE: open3d's OBJ reader cannot handle the non-ASCII export path
# (same limitation documented in 15_tsdf_from_depth.py), so we stage copies
# into an ASCII directory first.
OUT_DIR = Path(r"C:\isaacsim_data\mesh")

V1 = OUT_DIR / "gs_mesh_v1_for_compare.obj"
V2 = OUT_DIR / "gs_mesh_v2_for_compare.obj"


def stats(path):
    m = o3d.io.read_triangle_mesh(str(path))
    if len(m.triangles) == 0:
        return None
    tri = np.asarray(m.triangles)
    verts = np.asarray(m.vertices)

    # boundary edges: an edge shared by a single triangle = rim of a hole
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    boundary = int((cnt == 1).sum())

    cols = np.asarray(m.vertex_colors)
    chroma = float((cols.max(1) - cols.min(1)).mean() * 255) if len(cols) else float("nan")
    bright = float(cols.mean() * 255) if len(cols) else float("nan")

    bb = verts.max(0) - verts.min(0)
    return dict(
        name=path.name,
        verts=len(verts),
        tris=len(tri),
        boundary=boundary,
        chroma=chroma,
        bright=bright,
        bbox=f"{bb[0]:.3f} x {bb[1]:.3f} x {bb[2]:.3f}",
    )


rows = [r for r in (stats(V1), stats(V2)) if r]
if not rows:
    raise SystemExit("no meshes found")

hdr = f"{'mesh':<62}{'verts':>8}{'tris':>8}{'boundary':>10}{'chroma':>9}{'bright':>8}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['name'][:60]:<62}{r['verts']:>8}{r['tris']:>8}"
          f"{r['boundary']:>10}{r['chroma']:>9.1f}{r['bright']:>8.1f}")
print()
for r in rows:
    print(f"  {r['name']}\n    bbox = {r['bbox']} m")

if len(rows) == 2:
    a, b = rows
    print()
    print(f"  boundary edges : {a['boundary']} -> {b['boundary']}  "
          f"({(b['boundary']-a['boundary'])/max(a['boundary'],1)*100:+.0f}%)  lower = fewer holes")
    print(f"  colour chroma  : {a['chroma']:.1f} -> {b['chroma']:.1f}  "
          f"({(b['chroma']-a['chroma'])/max(a['chroma'],1e-6)*100:+.0f}%)  higher = more saturated")
