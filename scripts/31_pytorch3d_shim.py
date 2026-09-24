"""31_pytorch3d_shim.py -- Windows stand-in for the one pytorch3d call EcoSplat needs.

EcoSplat's `src/misc/cam_utils.py` imports `pytorch3d.transforms` but only ever
calls `rotation_6d_to_matrix()` (line 281). pytorch3d has no official Windows
wheel and building it from source is a multi-hour ordeal, so instead of porting
the scene to Linux we install a tiny shim package that provides that single
function with an identical signature and numerically identical result.

Run once:
    python 31_pytorch3d_shim.py

It writes <site-packages>/pytorch3d/{__init__,transforms}.py. Delete that folder
to undo.
"""

import os
import sys

TRANSFORMS = '''
"""Minimal Windows replacement for pytorch3d.transforms.

Only `rotation_6d_to_matrix` is implemented -- it is the only function EcoSplat
uses. Algorithm: Zhou et al., "On the Continuity of Rotation Representations in
Neural Networks" (CVPR 2019), identical to pytorch3d's implementation.
"""

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert a 6D rotation representation (..., 6) to a rotation matrix (..., 3, 3)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Inverse of rotation_6d_to_matrix: (..., 3, 3) -> (..., 6)."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))
'''

INIT = '''"""Windows shim package for pytorch3d (see scripts/31_pytorch3d_shim.py)."""

__version__ = "0.0.0-shim"
'''


def main():
    # find a writable site-packages for the *current* interpreter
    for cand in sys.path:
        if os.path.isdir(cand) and cand.lower().endswith("site-packages"):
            sp = cand
            break
    else:
        raise SystemExit("could not locate site-packages")

    pkg = os.path.join(sp, "pytorch3d")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(INIT)
    with open(os.path.join(pkg, "transforms.py"), "w", encoding="utf-8") as f:
        f.write(TRANSFORMS)

    print(f"shim installed at {pkg}")
    import torch
    from pytorch3d.transforms import rotation_6d_to_matrix
    m = rotation_6d_to_matrix(torch.tensor([[1.0, 0, 0, 0, 1.0, 0]]))
    print("self-test: identity ->\n", m)
    ok = torch.allclose(m @ m.transpose(-1, -2), torch.eye(3), atol=1e-5)
    print("orthonormal:", bool(ok))
    if not ok:
        raise SystemExit("shim self-test failed")


if __name__ == "__main__":
    main()
