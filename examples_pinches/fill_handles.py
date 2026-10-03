"""
fill_handles on the label image of the split_blocks.py example (run.py, with
remove_pinches; saved as examples_pinches/img_with_raw.vti, background 0), then
remove_pinches. Saves the repaired image, its surface nets and the membranes/loops.

Usage: python examples_pinches/fill_handles.py [workers]
"""

import sys
import time
from collections import Counter

import nbmorph
import numpy as np
import pyvista as pv

from emimesh.handles import components_per_label, fill_handles, labels_with_handles
from emimesh.pinches import count_pinches, remove_pinches


def to_world(img, ijk):
    return np.asarray(img.origin) + (np.asarray(ijk) + 0.5) * np.asarray(img.spacing)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    img = pv.read("examples_pinches/img_with_raw.vti")
    shape = np.array(img.dimensions) - 1
    d0 = img["data"].reshape(shape, order="F")
    nlabels = int(d0.max())
    t = time.time()
    d, fills = fill_handles(d0, workers=workers, return_fills=True)
    print(f"fill_handles: {time.time() - t:.1f}s")

    total = Counter()
    for *_, taken in fills:
        total.update(taken)
    print("voxels taken in total:", dict(total))
    b0, b1 = components_per_label(d0, nlabels), components_per_label(d, nlabels)
    print(
        "cells whose #components changed (before, after):",
        {lab: (int(b0[lab]), int(b1[lab])) for lab in np.nonzero(b0 != b1)[0]},
    )

    def contacts(x):
        return int((nbmorph.separate_labels_box(x) != x).sum())

    print("voxels touching another cell: before", contacts(d0), "after fill", contacts(d))
    print("pinched cubes after fill:", count_pinches(d))
    t = time.time()
    d = remove_pinches(d)
    print(f"remove_pinches: {time.time() - t:.1f}s")
    print("voxels touching another cell after remove_pinches:", contacts(d))
    print("cells with handles left:", labels_with_handles(d, nlabels).tolist())

    # membranes and loops for inspection
    blocks = pv.MultiBlock()
    for i, (lab, loop, grid, _) in enumerate(fills):
        blocks[f"loop_{i}"] = pv.lines_from_points(to_world(img, np.vstack([loop, loop[:1]])))
        if grid is None:  # cut handle
            continue
        g = to_world(img, grid)
        sg = pv.StructuredGrid(*[np.concatenate([g, g[:, :1]], axis=1)[..., c] for c in range(3)])
        sg["label"] = np.full(sg.n_points, lab)
        blocks[f"membrane_{i}"] = sg.extract_surface(algorithm="dataset_surface")
    blocks.combine().extract_surface(algorithm="dataset_surface").save(
        "examples_pinches/handle_membranes.vtk"
    )

    img["data"] = d.ravel(order="F")
    img.save("examples_pinches/img_with_fillhandles.vti")
    img["data"][img["data"] == 0] = 1  # same as run.py
    img.contour_labels("all", smoothing=True, background_value=0).save(
        "examples_pinches/surf_with_fillhandles.vtk"
    )
    print("saved")
