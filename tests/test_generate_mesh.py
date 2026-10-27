"""Tests for the curved (quadratic) meshes of emi-generate-mesh --curved True."""

import numpy as np
import pyvista as pv
from imagemesh.curved import compute_quadratic_quality
from imagemesh.image import np2pv
from imagemesh.surface import label_volumes
from imagemesh.tetmesh import compute_surface_volume, mesh_surface

from emimesh.clients.generate_mesh import curve_mesh
from emimesh.clients.generate_surface_tags import mark_interfaces
from emimesh.extract_surfaces import extract_surface

DX = 10.0


def test_curve_mesh():
    """A ball (label 2) in the ECS: snapping the quadratic mesh onto the smooth
    surface brings the ball volume closer to the volume the surface encloses."""
    z, y, x = np.mgrid[:16, :16, :16]
    img = np.where((x - 7.5) ** 2 + (y - 7.5) ** 2 + (z - 7.5) ** 2 < 5**2, 2, 0)
    surf = extract_surface(np2pv(img.astype(np.uint32), (DX,) * 3))
    mesh, _ = mesh_surface(surf, envelopsize=0.5 * DX, simplify_eps=0.5 * DX, edge_length_fac=0.1)
    quad = curve_mesh(mesh, surf)

    assert set(quad.celltypes) == {pv.CellType.QUADRATIC_TETRA}
    np.testing.assert_array_equal(quad["label"], mesh["label"])
    assert compute_quadratic_quality(quad).min() > 0

    target = label_volumes(surf.points, surf.regular_faces, surf["boundary_labels"])[2]
    linear_error = abs(compute_surface_volume(mesh, [2])[0][0] - target)
    curved_error = abs(compute_surface_volume(quad, [2])[0][0] - target)
    assert curved_error < 0.25 * linear_error

    facets = mark_interfaces(quad, outer_offset=10)
    assert set(facets.celltypes) == {pv.CellType.QUADRATIC_TRIANGLE}
    assert set(np.unique(facets["boundaries"])) == {2, 11}
