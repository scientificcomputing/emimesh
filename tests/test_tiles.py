"""Tile mode of emimesh.blocks.joint_surface and of process_image: a tile's
blocks plus a halo give the same surface as the global run."""

import itertools

import numpy as np
import pytest
from imagemesh.image import np2pv

from emimesh.blocks import joint_surface
from emimesh.process_image_data import process_image

DX = 16.0  # integer nm: the points are exact in any window
ORIGIN = np.array([32.0, 48.0, 64.0]) * DX
B = 20  # block size (voxels)
N = 3  # blocks per axis
OFF = 2**40  # global offset, above every body id
BODY = 10**9
FAIR = dict(influence_radius=12)  # 4 steps, reach 11 voxels
CAP_IT = 4
HALO = 12 + CAP_IT


def balls(shape=(N * B,) * 3, n=40, seed=1):
    """Overlapping balls with large body ids, on the grid's origin and spacing."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, np.int64)
    x = np.stack(np.meshgrid(*[np.arange(s) + 0.5 for s in shape], indexing="ij"), -1)
    for label in range(n):
        c = rng.uniform(0, 1, 3) * shape
        img[np.linalg.norm(x - c, axis=-1) < rng.uniform(4, 11)] = BODY + 2 + 3 * label
    return img


def grid(img, lo=(0, 0, 0)):
    g = np2pv(img, (DX,) * 3)
    g.origin = ORIGIN + np.asarray(lo) * DX
    return g


def window(core_lo, core_hi, halo=HALO, dims=(N * B,) * 3):
    """The voxel box of a tile's blocks [core_lo, core_hi) plus halo, clipped,
    its inner cuts (window voxel indices) and the block ids of its intervals."""
    lo = [max(0, B * c - halo) for c in core_lo]
    hi = [min(d, B * c + halo) for c, d in zip(core_hi, dims)]
    cuts, ids = [], []
    for a in range(3):
        planes = [B * i for i in range(1, N) if lo[a] < B * i < hi[a]]
        cuts.append([p - lo[a] for p in planes])
        ids.append(list(range(lo[a] // B, (hi[a] - 1) // B + 1)))
    return lo, hi, cuts, ids


def tile_surface(img, core_lo, core_hi, halo=HALO, **kw):
    lo, hi, cuts, ids = window(core_lo, core_hi, halo)
    sub = img[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    core = itertools.product(*[range(a, b) for a, b in zip(core_lo, core_hi)])
    return joint_surface(
        grid(sub, lo), cuts, offset=OFF, cap_iterations=CAP_IT, block_ids=ids,
        grid_n=(N, N, N), core=core, **FAIR, **kw,
    )  # fmt: skip


def crop(surf, core):
    """Faces of surf with a side in a core block (lin), the other side 0 if not."""
    tl = np.asarray(surf.cell_data["boundary_labels"], np.int64)
    tris = np.asarray(surf.regular_faces)
    lin = np.where(tl > 0, (tl - 1) // OFF, -1)
    inc = (tl > 0) & np.isin(lin, list(core))
    keep = inc.any(1)
    tris, tl = tris[keep], np.where(inc[keep], tl[keep], 0)
    turn = tl[:, 0] == 0
    tl[turn] = tl[turn][:, ::-1]
    tris[turn] = tris[turn][:, [0, 2, 1]]
    return np.asarray(surf.points)[tris], tl


def rows(P, tl):
    """One row per face: its labels and its corners from the smallest one,
    in cyclic order (orientation kept); sorted."""
    P = np.asarray(P, np.float64)
    out = []
    for tri in P:
        k = min(range(3), key=lambda i: tuple(tri[i]))
        out.append(np.roll(tri, -k, axis=0).ravel())
    r = np.column_stack([tl.astype(np.float64), np.array(out)])
    return r[np.lexsort(r.T[::-1])]


def lin(i, j, k):
    return i + N * (j + N * k)


@pytest.fixture(scope="module")
def image():
    return balls()


@pytest.fixture(scope="module")
def global_surface(image):
    cuts = [[B * i for i in range(1, N)]] * 3
    return joint_surface(
        grid(image), cuts, offset=OFF, simplify_eps=0, cap_iterations=CAP_IT, **FAIR
    )


@pytest.mark.parametrize("core", [(1, 1, 1), (0, 0, 0), (2, 1, 0)])
def test_tile_matches_global(image, global_surface, core):
    """With simplify_eps=0, a tile's faces (coordinates and labels) are bitwise
    those of the global surface in its block."""
    ijk = np.array(core)
    tile = tile_surface(image, ijk, ijk + 1, simplify_eps=0)
    assert tile.field_data["grid_n"].tolist() == [N] * 3
    assert tile.field_data["core"].tolist() == [lin(*core)]
    want = rows(*crop(global_surface, [lin(*core)]))
    tl = np.asarray(tile.cell_data["boundary_labels"], np.int64)
    got = rows(np.asarray(tile.points)[tile.regular_faces], tl)
    assert len(want) > 100
    # labels: lin off + body id, 0 outside the core
    assert ((tl[:, 0] - 1) // OFF == lin(*core)).all()
    assert ((tl[:, 1] == 0) | ((tl[:, 1] - 1) // OFF == lin(*core))).all()
    assert got.shape == want.shape, (got.shape, want.shape)
    diff = np.abs(got - want).max()
    assert (got == want).all(), f"largest difference {diff}"


def test_neighbour_tiles_planes(image):
    """Two neighbouring tiles, simplified with their inner grid planes fixed:
    the points on the shared plane are bitwise identical, and so are the cap
    faces on it."""
    inner = [(a, ORIGIN[a] + B * i * DX) for a in range(3) for i in range(1, N)]
    surf = {}
    for core in [(1, 1, 1), (2, 1, 1)]:
        ijk = np.array(core)
        sides = [
            (a, c)
            for a, c in inner
            if c in (ORIGIN[a] + ijk[a] * B * DX, ORIGIN[a] + (ijk[a] + 1) * B * DX)
        ]
        surf[core] = tile_surface(image, ijk, ijk + 1, simplify_eps=0.5 * DX, fixed_planes=sides)
    c = ORIGIN[0] + 2 * B * DX
    pa, pb = (np.asarray(s.points) for s in surf.values())
    on_a, on_b = pa[pa[:, 0] == c], pb[pb[:, 0] == c]
    assert len(on_a) > 50
    np.testing.assert_array_equal(np.unique(on_a, axis=0), np.unique(on_b, axis=0))

    def caps(s):
        P = np.asarray(s.points)[s.regular_faces]
        on = (P[:, :, 0] == c).all(1)
        tl = np.asarray(s.cell_data["boundary_labels"], np.int64)[on]
        assert (tl[:, 1] == 0).all()
        # each triangle as its sorted corners (the orientation is the block's)
        tri = np.array([sorted(map(tuple, t)) for t in P[on]]).reshape(-1, 9)
        return np.unique(tri, axis=0), (tl[:, 0] - 1) % OFF + 1

    (ta, la), (tb, lb) = caps(surf[(1, 1, 1)]), caps(surf[(2, 1, 1)])
    assert len(ta) > 20
    np.testing.assert_array_equal(ta, tb)
    np.testing.assert_array_equal(np.sort(np.unique(la)), np.sort(np.unique(lb)))
    # simplified away from the planes
    full = tile_surface(image, (1, 1, 1), (2, 2, 2), simplify_eps=0)
    assert surf[(1, 1, 1)].n_cells < full.n_cells


def test_tile_offset_checks(image):
    lo, hi, cuts, ids = window((1, 1, 1), (2, 2, 2))
    sub = grid(image[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], lo)
    with pytest.raises(ValueError, match="offset"):
        joint_surface(sub, cuts, block_ids=ids, grid_n=(N,) * 3, **FAIR)
    with pytest.raises(ValueError, match="exceed"):
        joint_surface(sub, cuts, offset=BODY, block_ids=ids, grid_n=(N,) * 3, **FAIR)
    with pytest.raises(ValueError, match="overflow"):
        joint_surface(sub, cuts, offset=2**61, block_ids=ids, grid_n=(N,) * 3, **FAIR)


OPS = [
    "removeislands minsize=500",
    "smooth iterations=1 radius=2",
    "dilate radius=1",
    "erode radius=1 struct_sequence=D",
    "prune_tips",
    "remove_pinches",
    "removeislands minsize=10",
]


def cells(shape=(64, 60, 56), n=60, seed=3):
    """A dense, noisy segmentation: nearest of n seeds (anisotropic distance),
    with salt noise, holes and large body ids."""
    rng = np.random.default_rng(seed)
    x = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    seeds = rng.uniform(0, 1, (n, 3)) * shape
    scale = rng.uniform(0.5, 2.0, (n, 3))
    d = np.stack([np.linalg.norm((x - s) * w, axis=-1) for s, w in zip(seeds, scale)])
    img = BODY + 2 + 7 * np.argmin(d, axis=0)
    img[np.sort(d, axis=0)[1] - np.min(d, axis=0) < 1.5] = 0  # gaps between cells
    salt = rng.random(shape) < 0.02
    img[salt] = BODY + 2 + 7 * rng.integers(0, n, salt.sum())
    img[rng.random(shape) < 0.02] = 0
    return img.astype(np.uint64)


def test_process_image_keep_ids():
    """keep_ids: body ids out (int64), not count-sorted labels; label 1 raises."""
    img = cells((24, 22, 20), n=8)
    out, cmap, labels, counts = process_image(
        grid(img), DX, [o.split() for o in OPS], keep_ids=True
    )
    data = np.asarray(out["data"])
    assert data.dtype == np.int64
    assert set(np.unique(data)) - {0} <= set(np.unique(img).tolist())
    assert set(labels) == set(np.unique(data).tolist()) - {0}
    assert all(cmap[k] == k for k in cmap)
    ref, *_ = process_image(grid(img), DX, [o.split() for o in OPS])
    # the same partition as the default output (only the labels differ)
    pairs = np.unique(np.column_stack([data, np.asarray(ref["data"])]), axis=0)
    assert len(pairs) == len(np.unique(data)) == len(np.unique(ref["data"]))
    img[0, 0, 0] = 1
    with pytest.raises(ValueError, match="ECS"):
        process_image(grid(img), DX, [o.split() for o in OPS], keep_ids=True)


RECONNECT_OPS = OPS[:5] + ["reconnect s=6 t=1"] + OPS[5:]


@pytest.mark.parametrize(
    "ops, halo, shape, lo, hi",
    [
        (OPS, 16, (64, 60, 56), (10, 4, 20), (50, 40, 56)),
        (RECONNECT_OPS, 30, (100, 96, 84), (13, 41, 7), (91, 96, 80)),
    ],
)
def test_process_image_window(ops, halo, shape, lo, hi):
    """The operations of the blockwise config on a window agree with the
    whole image in the window's core (keep_boundary_islands, keep_ids); the
    window's position comes from the grid origin."""
    img = cells(shape, n=int(60 * np.prod(shape) / (64 * 60 * 56)))
    ops = [o.split() for o in ops]
    kw = dict(keep_ids=True, keep_boundary_islands=True)
    full = np.asarray(process_image(grid(img), DX, ops, **kw)[0]["data"])
    full = full.reshape(img.shape, order="F")
    sub = img[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    win = np.asarray(process_image(grid(sub, lo), DX, ops, **kw)[0]["data"])
    win = win.reshape(sub.shape, order="F")
    # core: halo voxels from the window's sides inside the image
    core = tuple(
        slice(0 if a == 0 else halo, b - a if b == d else b - a - halo)
        for a, b, d in zip(lo, hi, img.shape)
    )
    ref = full[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]][core]
    assert (win[core] == ref).all(), f"{(win[core] != ref).sum()} voxels differ"
