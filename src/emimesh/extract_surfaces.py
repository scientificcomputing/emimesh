"""
Multi-label surface of the processed label image: surface nets
(contour_labels without smoothing) followed by our constrained smoothing,
which keeps the points on the bounding box on their box face.
"""

import numpy as np
import pyvista as pv

from emimesh.surface_smoothing import smooth_surface_net

ECS_LABEL = 1


def prepare_labels(imggrid):
    """
    Label image for the surface extraction: cells keep their labels (>= 2),
    the background becomes the ECS (label 1). If the image has a roimask,
    only the background inside the ROI becomes ECS, the rest stays 0
    (outside of the domain).
    """
    data = np.array(imggrid.cell_data["data"])
    ecs = data == 0
    if "roimask" in imggrid.array_names:
        ecs &= np.asarray(imggrid["roimask"]).astype(bool)
    data[ecs] = ECS_LABEL
    grid = pv.ImageData(
        dimensions=imggrid.dimensions, spacing=imggrid.spacing, origin=imggrid.origin
    )
    grid.cell_data["data"] = data
    return grid


def extract_surface(imggrid, smoothing_scale=1.2, iterations=16):
    """
    Smoothed, triangulated multi-label surface of the label image (cell data
    'data', 0 = outside), with cell data 'boundary_labels' (n_faces, 2).
    The displacement of each point is limited to smoothing_scale * dx; dx is
    stored in the field data.
    """
    grid = prepare_labels(imggrid)
    surf = grid.contour_labels(
        "all", smoothing=False, output_mesh_type="quads", background_value=0, scalars="data"
    )
    dx = np.min(grid.spacing)
    surf = smooth_surface_net(
        surf, iterations=iterations, distance=dx, scale=smoothing_scale, fix_bounds=True
    )
    surf.field_data["dx"] = [dx]
    return surf
