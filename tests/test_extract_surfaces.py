"""Tests for emimesh.extract_surfaces and emimesh.generate_mesh."""

import numpy as np
import pytest

from emimesh.ecs_share import label_volumes as surface_volumes
from emimesh.extract_surfaces import ECS_LABEL, extract_surface, prepare_labels
from emimesh.generate_mesh import mesh_surface
from emimesh.utils import np2pv

DX = 10.0


def two_cells(roi=False):
    """Two touching boxes (labels 2 and 3) in a 20^3 voxel image; with roi, a
    roimask excluding the outer 2 voxel layers in x."""
    img = np.zeros((20, 20, 20), dtype=np.uint32)
    img[4:10, 5:15, 5:15] = 2
    img[10:16, 5:15, 5:15] = 3
    roimask = None
    if roi:
        roimask = np.zeros(img.shape, dtype=np.uint8)
        roimask[2:18] = 1
    return np2pv(img, (DX,) * 3, roimask=roimask)


def label_volumes(mesh):
    vol = mesh.compute_cell_sizes(length=False, area=False)["Volume"]
    return {int(k): vol[mesh["label"] == k].sum() for k in np.unique(mesh["label"])}


def test_prepare_labels():
    grid = prepare_labels(two_cells())
    assert set(np.unique(grid["data"])) == {ECS_LABEL, 2, 3}

    grid = prepare_labels(two_cells(roi=True))
    data = grid["data"].reshape(np.array(grid.dimensions) - 1, order="F")
    assert np.all(data[:2] == 0) and np.all(data[18:] == 0)
    assert np.all(data[2:4] == ECS_LABEL)


@pytest.mark.parametrize("roi", [False, True])
def test_extract_surface(roi):
    imggrid = two_cells(roi)
    surf = extract_surface(imggrid)
    assert surf.is_all_triangles
    assert surf.field_data["dx"][0] == DX

    labels = surf.cell_data["boundary_labels"]
    pairs = {tuple(sorted(p)) for p in labels}
    assert {(1, 2), (1, 3), (2, 3)} <= pairs
    # the outside (0) only borders the ECS
    assert all(p[1] == ECS_LABEL for p in pairs if p[0] == 0)

    # each label region is closed: every edge of its faces is used twice
    for label in (1, 2, 3):
        faces = surf.regular_faces[(labels == label).any(axis=1)]
        edges = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], -1).reshape(-1, 2), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(counts == 2)

    # points stay inside the image, the box faces are kept
    lo, hi = np.array(imggrid.bounds).reshape(3, 2).T
    if roi:
        lo[0], hi[0] = 2 * DX, 18 * DX
    assert np.allclose(surf.points.min(axis=0), lo)
    assert np.allclose(surf.points.max(axis=0), hi)


@pytest.mark.parametrize("roi", [False, True])
def test_mesh_surface(roi):
    imggrid = two_cells(roi)
    surf = extract_surface(imggrid)
    mesh, dec = mesh_surface(surf, envelopsize=0.5, simplify_eps=0.05 * DX)
    assert dec.n_points <= surf.n_points

    vols = label_volumes(mesh)
    assert set(vols) == {1, 2, 3}
    # simplification, meshing and labeling keep the volumes enclosed by the surface
    ref = surface_volumes(
        surf.points.astype(float), surf.regular_faces, surf["boundary_labels"].astype(int)
    )
    for k in (1, 2, 3):
        assert vols[k] == pytest.approx(ref[k], rel=1e-2)
    # smoothing rounds off the edges of the thin boxes
    cell = 6 * 10 * 10 * DX**3
    assert vols[2] == pytest.approx(cell, rel=0.25)
    assert vols[3] == pytest.approx(cell, rel=0.25)
    box = (16 if roi else 20) * 20 * 20 * DX**3
    assert sum(vols.values()) == pytest.approx(box, rel=1e-3)
