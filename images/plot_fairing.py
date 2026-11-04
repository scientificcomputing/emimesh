"""
Laplacian smoothing and fairing of the surface of real cells: a 2 um cube of
the FlyEM male CNS segmentation (16 nm voxels), processed as in the readme
(opening, erosion for the ECS, prune_tips, remove_pinches). A close-up of one
cell: its voxel surface, and the surface nets smoothed with
surface: method: laplace (default) and surface: method: fair.

    python images/plot_fairing.py images/fairing.png [workdir]

The downloaded and processed image is cached in workdir (default:
results/fairing_figure).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pyvista as pv
from imagemesh.image import np2pv
from imagemesh.surface import label_volumes, surface_quads

from emimesh.download_data import download_cloudvolume
from emimesh.extract_surfaces import extract_surface, prepare_labels
from emimesh.process_image_data import process_image

warnings.simplefilter("ignore")
pv.OFF_SCREEN = True

URL = "precomputed://gs://flyem-male-cns/v1.0/segmentation"
POSITION = [48686, 27515, 24721]  # neuroglancer position, 8 nm voxels
DX = 16
OPS = [
    ["removeislands", "minsize=200"],
    ["smooth", "iterations=1", "radius=2"],
    ["dilate", "radius=1"],
    ["erode", "radius=1"],
    ["prune_tips"],
    ["remove_pinches"],
    ["removeislands", "minsize=200"],
]
CENTRE = np.array([389635.0, 219417.0, 197484.0])  # inside the cell shown (nm)
HALF = 170  # half width of the close-up box (nm)
DIRECTION = np.array([1.0, 1.0, 1.3]) / np.linalg.norm([1.0, 1.0, 1.3])
COLOUR = "#4f8fdf"


def processed_image(work):
    path = work / "processed.vti"
    if path.exists():
        return pv.read(path)
    img, res, origin = download_cloudvolume(
        URL, 1, POSITION, [2000] * 3, position_resolution=[8, 8, 8], return_origin=True
    )
    img, *_ = process_image(np2pv(img, res), dx=DX, operations=OPS)
    img.origin = origin
    work.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return img


def volume_change(surf, voxel_volume):
    """Relative volume change of every cell."""
    bl = np.asarray(surf.cell_data["boundary_labels"]).astype(np.int64)
    vol = label_volumes(np.asarray(surf.points), surf.regular_faces, bl)
    return {k: vol[k] / v - 1 for k, v in voxel_volume.items()}


def cell_surface(surf, label):
    bl = np.asarray(surf.cell_data["boundary_labels"])
    cell = surf.extract_cells(np.flatnonzero((bl == label).any(axis=1)))
    cell = cell.extract_surface(algorithm="dataset_surface")
    box = np.column_stack([CENTRE - HALF, CENTRE + HALF]).ravel()
    return cell.clip_box(box, invert=False)


def main(out, work):
    img = processed_image(Path(work))
    grid = prepare_labels(img)
    data = np.asarray(img.cell_data["data"])
    labels, counts = np.unique(data[data > 0], return_counts=True)
    voxel_volume = dict(zip(labels, counts * float(DX) ** 3))
    voxels = surface_quads(grid).triangulate()
    columns = [
        ("voxel surface", voxels),
        ("surface nets (laplace)", extract_surface(img)),
        ("surface nets (fair)", extract_surface(img, method="fair")),
    ]
    label = int(data[img.find_containing_cell(CENTRE)])
    focus = np.reshape(cell_surface(voxels, label).bounds, (3, 2)).mean(axis=1)
    focus[2] += 0.12 * HALF  # room for the text above
    p = pv.Plotter(shape=(1, 3), window_size=(1500, 540), border=False)
    for c, (title, surf) in enumerate(columns):
        p.subplot(0, c)
        p.add_mesh(cell_surface(surf, label), color=COLOUR, specular=0.3)
        if c:
            title += f"\nvolume change {volume_change(surf, voxel_volume)[label]:+.1%}"
        text = p.add_text(title, position=(0.02, 0.97), viewport=True, font_size=18)
        text.GetTextProperty().SetVerticalJustificationToTop()
        p.camera.focal_point = focus
        p.camera.position = focus + 3.2 * HALF * DIRECTION
        p.camera.up = (0, 0, 1)
    p.screenshot(out)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "fairing.png",
        sys.argv[2] if len(sys.argv) > 2 else "results/fairing_figure",
    )
