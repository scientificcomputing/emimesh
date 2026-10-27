"""
remove_pinches and fill_handles on a small synthetic label image.

* cell 2 consists of two blocks that touch only along an edge (a pinch): surface
  nets gives a non-manifold surface there.
* cell 3 is a square ring (a handle).

The image is repaired as by the processing operations "fill_handles" and
"remove_pinches", and both versions and their surface nets are saved.

Usage: python examples/pinches_and_handles.py [outdir]
(the first run takes a few minutes to compile the numba kernels)
"""

import sys
from pathlib import Path

import numpy as np
from imagemesh.handles import fill_handles, labels_with_handles
from imagemesh.image import np2pv
from imagemesh.pinches import count_pinches, remove_pinches

from emimesh.extract_surfaces import extract_surface


def make_image(n=48):
    img = np.zeros((n, n, n), np.uint16)
    # two blocks of cell 2, touching along the edge x = y = 12
    img[4:12, 4:12, 4:20] = 2
    img[12:20, 12:20, 4:20] = 2
    # square ring of cell 3 around the hole [30, 38) x [30, 38)
    img[26:42, 26:42, 26:32] = 3
    img[30:38, 30:38, 26:32] = 0
    return img


def report(name, img, outdir):
    print(
        f"{name}: {count_pinches(img)} pinched cubes, "
        f"cells with handles {labels_with_handles(img, int(img.max())).tolist()}"
    )
    grid = np2pv(img, [1.0] * 3)
    grid.save(outdir / f"img_{name}.vti")
    extract_surface(grid).save(outdir / f"surf_{name}.vtk")


if __name__ == "__main__":
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/examples")
    outdir.mkdir(parents=True, exist_ok=True)
    img = make_image()
    report("input", img, outdir)
    img = remove_pinches(fill_handles(img))
    report("repaired", img, outdir)
