"""32_build_ecosplat_dataset.py -- convert the Isaac Sim YCB capture into EcoSplat's
RealEstate10K-style chunk format.

EcoSplat's RE10K dataset reader (src/dataset/dataset_re10k.py) expects .torch files
holding a list of examples:

    {
      "key":     scene id (str),
      "images":  list of uint8 tensors, each the raw bytes of a JPEG/PNG file,
      "cameras": Float[Tensor, "n 18"]
                   [0:4]  = normalized intrinsics fx, fy, cx, cy
                            (pixelSplat convention: row0 / W, row1 / H)
                   [4:6]  = unused
                   [6:18] = OpenCV-style world-to-camera 3x4
    }

plus an index.json mapping scene key -> chunk filename.

Because EcoSplat is pose-free, the poses are only used as the *reference* frame
for view sampling and for scoring -- they are not fed to the network.
"""

import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGES = r"C:\Users\theozhu\Desktop\拓展\3DGS\data\dust3r_input"
GT_JSON = r"C:\Users\theozhu\Desktop\拓展\3DGS\data\ycb\transforms.json"
OUT_ROOT = r"C:\Users\theozhu\Desktop\拓展\Real2Sim-Grasping\ecoSplat\datasets\ycb\test"
SCENE_KEY = "ycb_cracker_box"
JPEG_QUALITY = 95


def main():
    gt = json.load(open(GT_JSON, encoding="utf-8"))
    W, H = int(gt["w"]), int(gt["h"])
    fx, fy = float(gt["fl_x"]), float(gt["fl_y"])
    cx, cy = float(gt["cx"]), float(gt["cy"])
    print(f"intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx} cy={cy}  ({W}x{H})")

    # pixelSplat normalisation: first row / W, second row / H
    fx_n, fy_n, cx_n, cy_n = fx / W, fy / H, cx / W, cy / H
    print(f"normalized : fx={fx_n:.4f} fy={fy_n:.4f} cx={cx_n:.4f} cy={cy_n:.4f}")

    # OpenGL (USD, nerfstudio) c2w -> OpenCV c2w: flip Y and Z axes
    flip = np.eye(4)
    flip[1, 1] = flip[2, 2] = -1.0

    images, cams = [], []
    for frame in gt["frames"]:
        name = frame["file_path"].split("/")[-1]
        path = os.path.join(IMAGES, name)
        if not os.path.exists(path):
            print("  missing, skip:", name)
            continue

        buf = BytesIO()
        Image.open(path).convert("RGB").save(buf, "JPEG", quality=JPEG_QUALITY)
        images.append(torch.from_numpy(
            np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()))

        c2w_gl = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w_cv = c2w_gl @ flip
        w2c = np.linalg.inv(c2w_cv)[:3, :4].astype(np.float32).reshape(-1)  # 12
        cam = np.zeros(18, dtype=np.float32)
        cam[0:4] = (fx_n, fy_n, cx_n, cy_n)
        cam[6:18] = w2c
        cams.append(cam)

    n = len(images)
    assert n > 0, "no images converted"
    example = {
        "key": SCENE_KEY,
        "images": images,
        "cameras": torch.stack([torch.from_numpy(c) for c in cams]),
    }
    print(f"converted {n} frames")

    os.makedirs(OUT_ROOT, exist_ok=True)
    chunk_name = "000000.torch"
    torch.save([example], os.path.join(OUT_ROOT, chunk_name))
    with open(os.path.join(OUT_ROOT, "index.json"), "w", encoding="utf-8") as f:
        json.dump({SCENE_KEY: chunk_name}, f, indent=2)

    size = os.path.getsize(os.path.join(OUT_ROOT, chunk_name)) / 1e6
    print(f"wrote {OUT_ROOT}\\{chunk_name}  ({size:.1f} MB)")
    print("wrote index.json:", {SCENE_KEY: chunk_name})


if __name__ == "__main__":
    main()
