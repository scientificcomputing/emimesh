"""Tests for emimesh.blocks."""

import numpy as np
import pyvista as pv

from emimesh.blocks import merge_surfaces


def test_merge_surfaces_only_cut_plane():
    """Points on the cut plane z = 1 are shared by the blocks; coincident points
    elsewhere (surface nets splits pinches into such points) stay separate."""
    a = pv.PolyData.from_regular_faces(
        np.array([[0, 0, 0.5], [1, 0, 0.5], [1, 0, 0.5], [0, 0, 1.0]]), np.array([[0, 1, 2, 3]])
    )
    b = pv.PolyData.from_regular_faces(
        np.array([[0, 0, 1.0], [1, 0, 1.5], [1, 0, 1.5], [0, 1, 1.5]]), np.array([[0, 1, 2, 3]])
    )
    points, (qa, qb) = merge_surfaces([a, b], 1e-3, [1.0])
    assert len(points) == 7
    assert qa[0, 3] == qb[0, 0]
    assert len(np.unique(qa)) == 4 and len(np.unique(qb)) == 4
