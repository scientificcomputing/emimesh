"""
The close-up of plot_fairing.py, faired with different ranges of influence
(surface: influence_radius): no point of the surface depends on a voxel
further away than this many voxels. The default has no limit.

    python images/plot_fairing_range.py images/fairing_range.png [workdir]

The processed image is cached in workdir (default: results/fairing_figure),
shared with plot_fairing.py.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from imagemesh.surface import surface_quads

sys.path.insert(0, str(Path(__file__).parent))
import plot_fairing as pf  # noqa: E402

from emimesh.extract_surfaces import extract_surface, prepare_labels  # noqa: E402

RANGES = [None, 10, 50, 100, 200]


def main(out, work):
    img = pf.processed_image(Path(work))
    grid = prepare_labels(img)
    data = np.asarray(img.cell_data["data"])
    labels, counts = np.unique(data[data > 0], return_counts=True)
    voxel_volume = dict(zip(labels, counts * float(pf.DX) ** 3))
    voxels = surface_quads(grid).triangulate()
    extract_surface(img, method="fair", influence_radius=10)  # compile
    columns = [("voxel surface", voxels)]
    for r in RANGES:
        t = time.perf_counter()
        surf = extract_surface(img, method="fair", influence_radius=r)
        name = "no limit (default)" if r is None else f"range {r} voxels"
        columns.append((f"fair, {name} ({time.perf_counter() - t:.0f} s)", surf))
    label = int(data[img.find_containing_cell(pf.CENTRE)])
    focus = np.reshape(pf.cell_surface(voxels, label).bounds, (3, 2)).mean(axis=1)
    focus[2] += 0.12 * pf.HALF  # room for the text above
    p = pv.Plotter(shape=(2, 3), window_size=(1500, 1080), border=False)
    for c, (title, surf) in enumerate(columns):
        p.subplot(c // 3, c % 3)
        p.add_mesh(pf.cell_surface(surf, label), color=pf.COLOUR, specular=0.3)
        if c:
            title += f"\nvolume change {pf.volume_change(surf, voxel_volume)[label]:+.1%}"
        text = p.add_text(title, position=(0.02, 0.97), viewport=True, font_size=16)
        text.GetTextProperty().SetVerticalJustificationToTop()
        p.camera.focal_point = focus
        p.camera.position = focus + 3.2 * pf.HALF * pf.DIRECTION
        p.camera.up = (0, 0, 1)
    p.screenshot(out)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "fairing_range.png",
        sys.argv[2] if len(sys.argv) > 2 else "results/fairing_figure",
    )
