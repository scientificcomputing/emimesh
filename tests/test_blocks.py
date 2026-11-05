"""Tests for emimesh.blocks."""

import numpy as np
import pytest
import pyvista as pv
from imagemesh.image import np2pv

from emimesh.blocks import joint_surface, merge_surfaces

DX = 10.0


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


def blobs(shape=(32, 30, 28), seed=0):
    """A few overlapping balls (labels 2..6) in a box."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, np.uint32)
    x = np.stack(np.meshgrid(*[np.arange(s) + 0.5 for s in shape], indexing="ij"), -1)
    for label in range(2, 7):
        c = rng.uniform(0.25, 0.75, 3) * shape
        img[np.linalg.norm(x - c, axis=-1) < rng.uniform(5, 9)] = label
    return np2pv(img, (DX,) * 3)


@pytest.fixture(scope="module", params=[2, (2, 1, 3)])
def joint(request):
    cuts = request.param
    n = (cuts,) * 3 if isinstance(cuts, int) else cuts
    return joint_surface(blobs(), cuts, influence_radius=30), n


def test_joint_surface_caps(joint):
    """The caps of a 2x2x2 (or 2x1x3) grid lie exactly on their planes."""
    surf, n = joint
    assert tuple(surf.field_data["block_n"]) == n
    planes = np.split(surf.field_data["block_cuts"], np.cumsum(np.add(n, 1))[:-1])
    off = int(surf.field_data["offset"][0])
    bl = surf.cell_data["boundary_labels"].astype(np.int64)
    base = np.where(bl > 0, (bl - 1) % off + 1, 0)
    cap = (base[:, 0] == base[:, 1]) & (bl[:, 0] != bl[:, 1])
    assert cap.any()
    for tri in surf.points[surf.regular_faces[cap]]:
        assert any(
            (tri[:, a] == tri[0, a]).all() and tri[0, a] in planes[a][1:-1] for a in range(3)
        )


def test_joint_surface_blocks(joint):
    """Every block surface cut from the joint surface passes tetopt's block
    contract (closed per label, inside its box, caps only on its seams)."""
    blockwise = pytest.importorskip("tetopt.blockwise.surfaces")
    surf, n = joint
    grid, blocks = blockwise.blocks_from_joint_surface(surf)
    assert grid.n == n and len(blocks) == np.prod(n)
    for ijk, block in blocks.items():
        assert blockwise.surface_problems_block(grid, ijk, block) == [], ijk
