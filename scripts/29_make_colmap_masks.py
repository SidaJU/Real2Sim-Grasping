"""Generate object masks from RGB only (no simulation ground truth).

The object (Cheez-It box) is saturated red/yellow; table and background are
gray/white. So the mask is a chroma threshold, NOT a brightness threshold -
the table measures ~207-214 brightness and would survive any "not white" rule.

    mask = (max(R,G,B) - min(R,G,B)) > CHROMA_THRESH

Outputs 8-bit masks next to the images, which nerfstudio reads via
--masks-path masks (splatfacto zeroes both GT and render outside the mask,
so background gaussians receive no gradient).
"""
import os

import numpy as np
from PIL import Image

IMAGES = r"C:\ycb_sfm\gs\images"
MASKS = r"C:\ycb_sfm\gs\masks"
CHROMA_THRESH = 40     # max(RGB)-min(RGB) above this = coloured object


def main():
    os.makedirs(MASKS, exist_ok=True)
    files = sorted(f for f in os.listdir(IMAGES)
                  if f.lower().endswith((".png", ".jpg")))
    ratios = []
    for fn in files:
        a = np.asarray(Image.open(os.path.join(IMAGES, fn)).convert("RGB"))
        chroma = a.max(2).astype(np.int16) - a.min(2).astype(np.int16)
        m = chroma > CHROMA_THRESH
        # The box's own white print (CHEEZ-IT lettering background) is gray too,
        # so chroma alone leaves holes INSIDE the object. Those pixels would get
        # no loss and the box surface would reconstruct with holes. Fill any
        # background region fully enclosed by the object silhouette.
        try:
            from scipy.ndimage import binary_fill_holes
            m = binary_fill_holes(m)
        except ImportError:
            pass
        m = m.astype(np.uint8) * 255
        Image.fromarray(m).save(os.path.join(MASKS, fn))
        ratios.append(100.0 * m.mean() / 255.0)

    ratios = np.array(ratios)
    print(f"masks written: {len(files)} -> {MASKS}")
    print(f"object coverage per image: min {ratios.min():.1f}%  "
          f"mean {ratios.mean():.1f}%  max {ratios.max():.1f}%")
    if ratios.min() < 1.0:
        print("WARNING: some images have almost no object pixels - "
              "check CHROMA_THRESH")
    if ratios.mean() > 60:
        print("WARNING: masks cover most of the frame - threshold too loose")


if __name__ == "__main__":
    main()
