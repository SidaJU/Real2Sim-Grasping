"""T3a  Dump gaussian centres + colours straight from the checkpoint.

Why not `ns-export gaussian-splat`?
    nerfstudio 1.1.5's exporter hits splatfacto.get_outputs() with a non-Cameras
    argument, which returns {} (splatfacto.py:539) -> "Could not find rgb in the
    model outputs". So we bypass the exporter entirely: the gaussian parameters
    are just tensors in the checkpoint.

Bonus: because we read the raw parameters, we can filter floaters properly
using OPACITY and SCALE - much more reliable than geometric outlier removal.

Output: gaussians.ply  (x y z + rgb, in nerfstudio's NORMALISED space)
"""

import sys
import os
import torch
import numpy as np

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    r"C:\Users\theozhu\Desktop\拓展\3DGS\outputs\ycb\splatfacto"
    r"\2026-09-16_104130\nerfstudio_models\step-000029999.ckpt"
)
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\theozhu\Desktop\拓展\3DGS\exports\ycb"

SH_C0 = 0.28209479177387814

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["pipeline"]

# key prefix differs between versions ("_model." vs "model.") -> auto-detect
def find(suffix):
    hits = [k for k in sd if k.endswith(suffix)]
    if not hits:
        raise SystemExit(f"key not found: {suffix}\navailable (sample): "
                         f"{[k for k in sd][:8]}")
    return hits[0]

k_means = find("gauss_params.means")
k_scales = find("gauss_params.scales")
k_op = find("gauss_params.opacities")
k_dc = find("gauss_params.features_dc")
print("keys:", k_means, "|", k_scales, "|", k_op, "|", k_dc)

means = sd[k_means].detach().cpu().numpy()
scales = torch.exp(sd[k_scales]).detach().cpu().numpy()      # exp() -> world units
opacity = torch.sigmoid(sd[k_op]).detach().cpu().numpy().reshape(-1)
fdc = sd[k_dc].detach().cpu().numpy()                        # (N, 1, 3)
rgb = np.clip(0.5 + SH_C0 * fdc.reshape(-1, 3), 0.0, 1.0)

print(f"gaussians: {len(means)}")

# ---- filter floaters: low opacity or oversized gaussians ----
max_scale = scales.max(axis=1)
op_thresh = 0.5
scale_thresh = float(np.quantile(max_scale, 0.90))
keep = (opacity > op_thresh) & (max_scale < scale_thresh)
print(f"opencv thresh: opacity>{op_thresh}, max_scale<{scale_thresh:.4f} (p90)")
print(f"kept {int(keep.sum())} / {len(means)} gaussians")

means, rgb = means[keep], rgb[keep]

os.makedirs(OUT, exist_ok=True)
ply = os.path.join(OUT, "gaussians.ply")
with open(ply, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(means)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")
    for (x, y, z), (r, g, b) in zip(means, rgb):
        f.write(f"{x:.6f} {y:.6f} {z:.6f} "
                f"{int(r*255)} {int(g*255)} {int(b*255)}\n")
print("saved:", ply)
