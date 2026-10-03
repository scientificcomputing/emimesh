"""
Benchmark fill_handles: tile the example n x n x n times (distinct labels per tile),
for a larger volume with the same feature sizes.

Usage: python examples_pinches/bench_tiles.py n workers
"""

import resource
import sys
import time

import numpy as np
import pyvista as pv

from emimesh.handles import fill_handles
from emimesh.pinches import remove_pinches

if __name__ == "__main__":
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    img = pv.read("examples_pinches/img_with_raw.vti")
    d = img["data"].reshape(np.array(img.dimensions) - 1, order="F").astype(np.uint16)
    nl = int(d.max())
    s = d.shape
    big = np.zeros((n * s[0], n * s[1], n * s[2]), np.uint16)
    for t, (i, j, k) in enumerate(np.ndindex(n, n, n)):
        big[i * s[0] : (i + 1) * s[0], j * s[1] : (j + 1) * s[1], k * s[2] : (k + 1) * s[2]] = (
            np.where(d > 0, d + t * nl, 0)
        )
    del d
    print("shape", big.shape, f"{big.size / 1e6:.0f}M voxels, {int(big.max())} cells", flush=True)
    fill_handles(big[:40, :40, :40], workers=1)  # numba/nbmorph compilation
    t = time.time()
    out, fills = fill_handles(big, workers=workers, return_fills=True)
    print(f"fill_handles {time.time() - t:.0f}s, {len(fills)} fills", flush=True)
    t = time.time()
    remove_pinches(out)
    print(f"remove_pinches {time.time() - t:.0f}s")
    print(
        f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB (main process)"
    )
