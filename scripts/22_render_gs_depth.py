"""T5a  Render depth maps FROM THE TRAINED 3DGS MODEL.

This is the missing link that closes the Real2Sim loop: the model looks at the
scene through the training cameras and tells us how far it thinks each pixel is.
No ground-truth depth involved - these depths are INFERRED from RGB only.

Why not `ns-export`?
    splatfacto.get_outputs() requires a `Cameras` object (splatfacto.py:539
    returns {} otherwise), but ns-export / ns-render still pass a RayBundle
    (nerfstudio/scripts/render.py:166). Version 1.1.5 is the latest on PyPI,
    so we do not wait for upstream - we build the Cameras ourselves.

Run with the nerfstudio env (NOT Isaac Sim):
    C:\\Users\\theozhu\\miniconda3\\envs\\nerfstudio\\python.exe 22_render_gs_depth.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup

CONFIG = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\outputs\ycb\splatfacto"
              r"\2026-09-23_102733\config.yml")   # GT poses + true-depth init, 30k steps
                                                  # (the run whose splat.ply we exported)
DATA = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb\transforms.json")
OUT = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\exports\ycb\gs_depth")

if not CONFIG.exists():
    sys.exit(f"config not found: {CONFIG}")

print("loading model (this takes ~30s)...")
config, pipeline, checkpoint_path, step = eval_setup(CONFIG)
model = pipeline.model
device = pipeline.device
print(f"loaded step {step}")

tf = json.load(open(DATA))
W, H, fx, fy, cx, cy = tf["w"], tf["h"], tf["fl_x"], tf["fl_y"], tf["cx"], tf["cy"]

OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*.npy"):
    f.unlink()

# IMPORTANT: the model lives in nerfstudio's NORMALISED space (auto-scale).
# Rendering with the raw real-world poses puts the cameras somewhere the model
# never saw -> every depth comes back 0.
#
# config.yml stores `data` as a RELATIVE path, so pipeline.datamanager may fail
# to resolve it here -> do NOT trust its cameras. Instead read the exact
# normalisation the dataparser applied, saved next to the config:
#   forward:  p_norm = T4 @ p_real ;  p_norm[:3,3] *= scale
# (inverse of nerfstudio's transform_poses_to_original_space, which divides by
#  scale first and then applies inv(T))
_np = CONFIG.parent / "dataparser_transforms.json"
_nrm = json.load(open(_np))
T4 = np.eye(4)
T4[:3, :4] = np.array(_nrm["transform"], dtype=np.float64)
SCALE = float(_nrm["scale"])
print(f"dataparser: translation {np.round(T4[:3, 3], 4)}  scale {SCALE:.4f}")

model.eval()
hit = 0
with torch.no_grad():
    for i, fr in enumerate(tf["frames"]):
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w_n = T4 @ c2w                    # real metres -> dataparser space
        c2w_n[:3, 3] *= SCALE
        cameras = Cameras(
            camera_to_worlds=torch.from_numpy(c2w_n[None, :3, :4].astype(np.float32)),
            fx=torch.tensor([[fx]], dtype=torch.float32),
            fy=torch.tensor([[fy]], dtype=torch.float32),
            cx=torch.tensor([[cx]], dtype=torch.float32),
            cy=torch.tensor([[cy]], dtype=torch.float32),
            width=torch.tensor([[W]], dtype=torch.int32),
            height=torch.tensor([[H]], dtype=torch.int32),
        ).to(device)

        outputs = model.get_outputs(cameras)
        if not outputs or "depth" not in outputs:
            print(f"frame {i}: NO DEPTH (keys: {list(outputs.keys())})")
            continue
        depth = outputs["depth"].detach().cpu().numpy().reshape(H, W)
        np.save(OUT / f"depth_{i:04d}.npy", depth)
        if depth.max() > 0:
            hit += 1
        if i % 15 == 0:
            print(f"frame {i:3d}: depth mean {depth.mean():.3f} "
                  f"max {depth.max():.3f} (normalised units)")

print(f"\nframes that saw geometry: {hit}/{len(tf['frames'])}")
print("next: 23_gs_to_mesh.py (TSDF + compare against the GT mesh)")
