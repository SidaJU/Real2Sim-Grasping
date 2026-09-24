"""T6c  Diagnose a trained splatfacto model: did it actually learn anything?

Checks:
  1. number of gaussians (vs the init cloud) - densify/cull happened?
  2. opacity distribution - are they solid or ghostly?
  3. render with the datamanager's own cameras - rgb / depth non-zero?
  4. render with real-world poses scaled into normalised space - sanity
"""

import sys
from pathlib import Path

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup

CONFIG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\Users\theozhu\Desktop\拓展\3DGS\outputs\ycb\splatfacto"
    r"\2026-09-17_204301\config.yml"
)
DATA = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb\transforms.json")

import json
tf = json.load(open(DATA))
W, H, fx, fy, cx, cy = tf["w"], tf["h"], tf["fl_x"], tf["fl_y"], tf["cx"], tf["cy"]

print("loading", CONFIG)
config, pipeline, ckpt_path, step = eval_setup(CONFIG)
model = pipeline.model
dev = pipeline.device

means = model.gauss_params["means"]
op = torch.sigmoid(model.gauss_params["opacities"]).detach().cpu().numpy()
print(f"\n[1] gaussians: {means.shape[0]}   (init cloud was 23785)")
print(f"    opacity: mean {op.mean():.3f}  median {np.median(op):.3f}  "
      f"max {op.max():.3f}  >0.5: {(op>0.5).mean()*100:.1f}%")

ds = pipeline.datamanager.train_dataset
print(f"\n[2] training cameras: {len(ds)}")
p = ds.cameras.camera_to_worlds.detach().cpu().numpy()
xy = p[:, :2, 3]
r_norm = float(np.linalg.norm(xy - xy.mean(0), axis=1).mean())
print(f"    orbit radius in normalised units: {r_norm:.4f}")
SCALE = r_norm / 0.25
print(f"    => space is {SCALE:.4f}x real world")

model.eval()
with torch.no_grad():
    # (a) datamanager camera
    cam_a = ds.cameras[0:1].to(dev)
    out_a = model.get_outputs(cam_a)
    rgb_a = out_a.get("rgb")
    dep_a = out_a.get("depth")
    print("\n[3] render with datamanager camera:")
    print(f"    keys: {list(out_a.keys())}")
    if rgb_a is not None:
        print(f"    rgb  mean {rgb_a.mean().item():.4f}  max {rgb_a.max().item():.4f}")
    if dep_a is not None:
        print(f"    depth mean {dep_a.mean().item():.4f}  max {dep_a.max().item():.4f}")

    # (b) real pose scaled into normalised space
    c2w = np.array(tf["frames"][0]["transform_matrix"], dtype=np.float32)
    c2w[:3, 3] *= SCALE
    cam_b = Cameras(
        camera_to_worlds=torch.from_numpy(c2w[None, :3, :4]),
        fx=torch.tensor([[fx]]), fy=torch.tensor([[fy]]),
        cx=torch.tensor([[cx]]), cy=torch.tensor([[cy]]),
        width=torch.tensor([[W]]), height=torch.tensor([[H]]),
    ).to(dev)
    out_b = model.get_outputs(cam_b)
    print("\n[4] render with scaled real-world camera:")
    if out_b.get("rgb") is not None:
        print(f"    rgb  mean {out_b['rgb'].mean().item():.4f}")
    if out_b.get("depth") is not None:
        print(f"    depth mean {out_b['depth'].mean().item():.4f}  "
              f"max {out_b['depth'].max().item():.4f}")

print("\n[5] model config: densify_until =",
      getattr(model.config, "densify_until_iter", "?"),
      " stop_split_at =", getattr(model.config, "stop_split_at", "?"))
