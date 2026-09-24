"""T1.3  Check exposure consistency across the 108 captures.

3DGS assumes every pixel of the same surface has the same colour in every view.
If Isaac Sim's auto-exposure drifted during capture, frames disagree and the
optimiser spawns per-camera 'floater' gaussians (the red ghost shards).

Healthy: mean brightness std < ~1.0 and no long drift.
"""

import os
import numpy as np
from PIL import Image

D = "C:/Users/theozhu/Desktop/拓展/3DGS/data/ycb/images"
files = sorted(f for f in os.listdir(D) if f.endswith(".png"))
means = []
for f in files:
    im = np.asarray(Image.open(os.path.join(D, f)).convert("L"), dtype=np.float32)
    means.append(im.mean())
m = np.array(means)
print(f"frames: {len(m)}")
print(f"brightness mean={m.mean():.2f}  std={m.std():.2f}  min={m.min():.2f}  max={m.max():.2f}")
print(f"max drift within one height ring:")
for r in range(3):
    seg = m[r * 36:(r + 1) * 36]
    print(f"  ring {r} (h={ [0.05,0.15,0.30][r] }): min={seg.min():.2f} max={seg.max():.2f} "
          f"drift={seg.max()-seg.min():.2f}")
print("first 12 frames:", np.round(m[:12], 2).tolist())
print("last  12 frames:", np.round(m[-12:], 2).tolist())
