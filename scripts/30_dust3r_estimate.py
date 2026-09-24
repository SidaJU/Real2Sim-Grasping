"""30_dust3r_estimate.py -- DUSt3R pose + dense pointcloud, scored against sim GT.

DUSt3R is a learning-based SfM alternative: it never matches SIFT features, so
it should NOT suffer the texture-repetition failure that pushed COLMAP to an
18.21 cm pose error on this scene. This script runs the official pipeline and
measures the result against the Isaac Sim ground truth, using the same Umeyama
alignment as 13_colmap_vs_gt.py so the numbers are directly comparable.

Usage:
    python 30_dust3r_estimate.py                 # 12 images, smoke test
    python 30_dust3r_estimate.py --n 99 --niter 300
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

DUST3R = r"C:\Users\theozhu\Desktop\拓展\Real2Sim-Grasping\dust3r"
sys.path.insert(0, DUST3R)

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner  # noqa: E402
from dust3r.image_pairs import make_pairs  # noqa: E402
from dust3r.inference import inference  # noqa: E402
from dust3r.model import AsymmetricCroCo3DStereo  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402

IMAGES = r"C:\Users\theozhu\Desktop\拓展\3DGS\data\dust3r_input"
GT_JSON = r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb\transforms.json"
OUT_DIR = r"C:\Users\theozhu\Desktop\拓展\Real2Sim-Grasping\results\dust3r"
MODEL = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"


def umeyama(src, dst):
    """Best-fit similarity transform (R, t, scale) mapping src -> dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / ((src - mu_s) ** 2).sum() * len(src)
    t = mu_d - scale * (R @ mu_s)
    return R, t, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12,
                    help="number of images to use (0 = all, evenly sampled)")
    ap.add_argument("--niter", type=int, default=300)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--graph", default="complete",
                    choices=["complete", "swin", "logwin", "oneref"])
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[1/5] loading {MODEL} ...")
    model = AsymmetricCroCo3DStereo.from_pretrained(MODEL).to(device)
    model.eval()

    files = sorted(f for f in os.listdir(IMAGES) if f.lower().endswith(".png"))
    keep = list(range(len(files)))
    if args.n > 0 and args.n < len(files):
        keep = sorted(set(int(round(i)) for i in np.linspace(0, len(files) - 1, args.n)))
    files = [files[i] for i in keep]
    print(f"[2/5] {len(files)} images, size={args.size}")

    # Official loader uses PIL.Image.open -> safe on non-ASCII (Chinese) paths.
    imgs = load_images([os.path.join(IMAGES, f) for f in files],
                       size=args.size, verbose=False)

    print(f"[3/5] pairwise inference (graph={args.graph}) ...")
    pairs = make_pairs(imgs, scene_graph=args.graph, prefilter=None, symmetrize=True)
    with torch.no_grad():
        out = inference(pairs, model, device, batch_size=args.batch)

    print(f"[4/5] global alignment ({args.niter} iters) ...")
    scene = global_aligner(out, device=device,
                           mode=GlobalAlignerMode.PointCloudOptimizer)
    # NOTE: no torch.no_grad() here -- the global alignment is a differentiable
    # optimisation over poses/depthmaps/focals and needs autograd. Wrapping it
    # raises "element 0 of tensors does not require grad".
    scene.compute_global_alignment(init="mst", niter=args.niter,
                                   schedule="cosine", lr=0.01)

    poses = scene.get_im_poses().detach().cpu().numpy()      # (N,4,4) cam->world
    intr = scene.get_intrinsics().detach().cpu().numpy()     # (N,3,3)
    pts3d = [p.detach().cpu().numpy() for p in scene.get_pts3d()]

    # ---- score against ground truth -------------------------------------
    gt = json.load(open(GT_JSON, encoding="utf-8"))
    gt_c = {}
    for fr in gt["frames"]:
        name = fr["file_path"].split("/")[-1]
        gt_c[name] = np.array(fr["transform_matrix"], dtype=np.float64)[:3, 3]

    est_c = poses[:, :3, 3]                       # camera centres, convention-free
    dst = np.array([gt_c[f] for f in files])

    R, t, scale = umeyama(est_c, dst)
    aligned = (scale * (R @ est_c.T).T) + t
    err = np.linalg.norm(aligned - dst, axis=1)

    print("\n=== DUSt3R vs simulation ground truth ===")
    print(f"  images            : {len(files)}")
    print(f"  scale factor      : {scale:.4f}")
    print(f"  position error    : mean {err.mean()*100:.2f} cm, "
          f"median {np.median(err)*100:.2f} cm, max {err.max()*100:.2f} cm")
    print(f"  focal (est)       : mean fx {intr[:,0,0].mean():.1f}, "
          f"fy {intr[:,1,1].mean():.1f}   (GT {gt.get('fl_x')})")
    npts = sum(p.shape[0] * p.shape[1] for p in pts3d)
    print(f"  dense pointmaps   : {len(pts3d)} maps, {npts/1e6:.2f} M points total")

    # ---- persist ---------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f"dust3r_{len(files)}img.npz"),
             names=np.array(files), poses=poses, intrinsics=intr,
             scale=scale, err_cm=err * 100)

    # export a COLMAP-style init cloud nerfstudio can consume
    allp, allc = [], []
    for i, p in enumerate(pts3d):
        m = np.isfinite(p).all(-1) & (np.linalg.norm(p, axis=-1) < 2.0)
        allp.append(p[m])
    cloud = np.concatenate(allp)
    if len(cloud) > 300000:
        cloud = cloud[np.random.RandomState(0).choice(len(cloud), 300000, replace=False)]
    cloud = (scale * (R @ cloud.T).T) + t
    ply = os.path.join(OUT_DIR, f"dust3r_init_{len(files)}img.ply")
    with open(ply, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(cloud)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        mid = np.full((len(cloud), 3), 128, np.uint8)
        for (x, y, z), (r, g, b) in zip(cloud.tolist(), mid.tolist()):
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")
    print(f"  saved: {ply}")
    print("[5/5] done.")


if __name__ == "__main__":
    main()
