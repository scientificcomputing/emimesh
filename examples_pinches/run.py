"""Surface nets of the split_blocks.py example, with and without remove_pinches."""
import time

import numpy as np
import pyvista as pv

from emimesh.pinches import count_pinches
from emimesh.process_image_data import process_image

ops = [
    ["removeislands", "minsize=5000"],
    ["dilate", "radius=1"],
    ["mode", "iterations=2"],
    ["smooth", "iterations=1", "radius=1"],
    ["upsample", "factor=2"],
    ["mode", "iterations=1"],
    ["zero_edges"],
    ["mode", "iterations=1"],
    ["zero_edges"],
]


def pinch_stats(surf):
    """Non-manifold edges (after merging coincident points) of a surface net."""
    surf = surf.clean(tolerance=1e-6)
    f = surf.regular_faces
    e = np.sort(np.stack([f, np.roll(f, -1, axis=1)], axis=2).reshape(-1, 2), axis=1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    return surf, ue[cnt != 2]


orig = pv.read("orig.vti")
for name, extra in [("without", []), ("with", [["remove_pinches"]])]:
    t = time.time()
    img, *_ = process_image(orig.copy(), dx=20, operations=ops + extra, ncells=100)
    print(f"{name}: processing {time.time() - t:.0f}s, image {img.dimensions}")
    data = img["data"].reshape(np.array(img.dimensions) - 1, order="F")
    print(f"{name}: pinched cubes {count_pinches(data)}")
    img["data"][img["data"] == 0] = 1
    raw = img.contour_labels("all", smoothing=False, output_mesh_type="quads", background_value=0)
    raw, bad = pinch_stats(raw.triangulate())
    print(f"{name}: non-manifold/open edges {len(bad)}")
    np.save(f"examples_pinches/bad_edges_{name}.npy", raw.points[bad].mean(axis=1))
    smooth = img.contour_labels("all", smoothing=True, background_value=0)
    smooth.save(f"examples_pinches/surf_{name}.vtk")
    img.save(f"examples_pinches/img_{name}.vti")
