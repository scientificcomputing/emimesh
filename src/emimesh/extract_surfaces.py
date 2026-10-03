"""
Multi-label surface of the processed label image: the background becomes the
ECS, then imagemesh's surface nets with constrained smoothing.
"""

import numpy as np
import pyvista as pv

from imagemesh.surface import extract_surface as extract_label_surface

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
    return extract_label_surface(
        prepare_labels(imggrid), smoothing_scale=smoothing_scale, iterations=iterations
    )
