"""T5c  Visualise the 3DGS-inferred depth next to the ground-truth depth.

Side-by-side, same colour scale, same frame - so any difference you see is
purely what the model got wrong (not a scaling or colour-mapping artefact).

Also prints per-frame stats and the median absolute error, which is the
cleanest single number for "how good are the inferred depths".
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GS_DEPTH = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\exports\ycb\gs_depth")
GT_DEPTH = Path(r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb\depth")   # rgb_XXXX.npy
OUT = Path(r"C:\isaacsim_data\mesh\gs_vs_gt_depth.png")

FRAMES = [0, 20, 40, 60, 80]
VMIN, VMAX = 0.05, 0.60      # metres, shared colour scale


def load_gs(i):
    return np.load(GS_DEPTH / f"depth_{i:04d}.npy").astype(np.float32)


def load_gt(i):
    return np.load(GT_DEPTH / f"rgb_{i:04d}.npy").astype(np.float32)


# calibrate: the model lives in nerfstudio's normalised space.
# Use ALL frames (not just the 5 we plot) - with only a few frames the
# estimate wobbles by ~10% because the inferred depth is noisy.
ratios = []
for i in range(0, 99, 2):
    try:
        gs, gt = load_gs(i), load_gt(i)
    except FileNotFoundError:
        continue
    m = np.isfinite(gs) & np.isfinite(gt) & (gt > 0.05) & (gs > 0.05)
    if m.sum() > 5000:
        ratios.append(float(np.median(gs[m]) / np.median(gt[m])))
SCALE = float(np.median(ratios))
print(f"normalised-unit scale: {SCALE:.4f}  (from {len(ratios)} frames)")

fig, axs = plt.subplots(len(FRAMES), 2, figsize=(10, 3.2 * len(FRAMES)))
for r, i in enumerate(FRAMES):
    gs = load_gs(i) / SCALE
    gt = load_gt(i)
    # rays that hit nothing are inf in the GT; blank them out instead of
    # letting them pollute the colour scale and the stats
    gs = np.where(np.isfinite(gs), gs, np.nan)
    gt = np.where(np.isfinite(gt), gt, np.nan)

    valid = np.isfinite(gs) & np.isfinite(gt) & (gt > 0.05) & (gs > 0.05)
    err = np.abs(gs[valid] - gt[valid]) if valid.sum() else np.array([np.nan])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    im0 = axs[r, 0].imshow(np.clip(gs, VMIN, VMAX), cmap=cmap, vmin=VMIN, vmax=VMAX)
    axs[r, 0].set_title(f"3DGS  frame {i}  mean {np.nanmean(gs):.3f} m")
    axs[r, 1].imshow(np.clip(gt, VMIN, VMAX), cmap=cmap, vmin=VMIN, vmax=VMAX)
    axs[r, 1].set_title(f"GT  frame {i}  mean {np.nanmean(gt):.3f} m  |  "
                        f"median err {np.nanmedian(err)*1000:.1f} mm")
    for c in (0, 1):
        axs[r, c].axis("off")
    print(f"frame {i:3d}: gs mean {np.nanmean(gs):.3f} m | gt mean {np.nanmean(gt):.3f} m "
          f"| median abs err {np.nanmedian(err)*1000:.1f} mm")

fig.colorbar(im0, ax=axs.ravel().tolist(), shrink=0.6, label="depth (m)")
fig.suptitle("3DGS-inferred depth (left) vs ground truth (right)", fontsize=13)
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("saved:", OUT)
