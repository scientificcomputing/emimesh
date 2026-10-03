"""
Tetrahedral mesh of a multi-label surface: simplification by edge collapses,
fTetWild (pytetwild) meshing of the whole surface and labeling of the
tetrahedra by the generalized winding number of the input surface.
"""

import time

import numpy as np
import pytetwild

from emimesh.surface_simplification import simplify_surface
from emimesh.winding_number import label_points

__all__ = ["mark_mesh", "mesh_surface"]


def mark_mesh(mesh, surf):
    """
    Label each tetrahedron of mesh with the region of surf (cell data
    'boundary_labels') containing its center, and drop the tetrahedra
    outside all regions (label 0).
    """
    marker = label_points(
        surf.points,
        surf.regular_faces,
        surf.cell_data["boundary_labels"],
        mesh.cell_centers().points,
    )
    mesh.cell_data["label"] = marker
    return mesh.extract_cells(marker > 0)


def mesh_surface(
    surf,
    envelopsize,
    simplify_eps=None,
    stop_quality=10,
    edge_length_fac=0.05,
    max_threads=1,
    quiet=True,
):
    """
    Simplify the multi-label surface surf (simplify_eps: absolute distance
    tolerance, None to skip) and tetrahedralize it with fTetWild.

    envelopsize: absolute size of the fTetWild surface envelope.
    edge_length_fac: target edge length relative to the bounding box diagonal.

    Returns the tet mesh with cell data 'label' and the simplified surface.
    """
    start = time.time()
    if simplify_eps:
        n = surf.n_points
        surf = simplify_surface(surf, epsilon=simplify_eps)
        print(f"simplified surface: {n} -> {surf.n_points} points")

    diag = np.linalg.norm(np.ptp(surf.points, axis=0))
    mesh = pytetwild.tetrahedralize_pv(
        surf,
        edge_length_fac=edge_length_fac,
        epsilon=envelopsize / diag,
        stop_energy=stop_quality,
        disable_filtering=True,
        coarsen=False,
        num_threads=max_threads,
        quiet=quiet,
    )
    mesh = mark_mesh(mesh, surf)
    print("meshing finished!")
    mesh.field_data["runtime"] = time.time() - start
    mesh.field_data["threads"] = max_threads
    return mesh, surf
