"""Helpers for meshing a label image in blocks (see split_blocks.py, split_grid.py)."""

import numpy as np
import pyvista as pv


def merge_surfaces(surfs, eps, planes):
    """
    Merge the points of the block surfaces that lie on the cut planes z in
    planes (coincident within eps). Returns the merged points and, per block,
    its quads in merged point ids. Other coincident points are not merged:
    surface nets splits pinches into coincident points, and merging them
    gives non-manifold edges whose points are locked in the smoothing.
    """
    pts = np.vstack([s.points for s in surfs]).astype(np.float64)
    key = np.round(pts / eps)
    off = ~np.isclose(pts[:, 2][:, None], planes, rtol=0, atol=eps).any(axis=1)
    key[off] = np.column_stack([np.zeros((off.sum(), 2)), -1 - np.arange(off.sum())])
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    offsets = np.cumsum([0] + [s.n_points for s in surfs])
    quads = [inv[s.regular_faces + o] for s, o in zip(surfs, offsets[:-1])]
    return pts[idx], quads


def _cut_indices(cuts, dims):
    """Voxel index of every cut plane per axis, 0 and dims included: cuts is
    the number of blocks (int, or one per axis; equal blocks, rounded) or
    the inner cut indices per axis."""
    if isinstance(cuts, (int, np.integer)):
        cuts = [int(cuts)] * 3
    if len(cuts) != 3:
        raise ValueError("cuts: an int, or one entry per axis")
    out = []
    for a, c in enumerate(cuts):
        if isinstance(c, (int, np.integer)):
            k = np.rint(np.arange(int(c) + 1) * dims[a] / int(c)).astype(int)
        else:
            k = np.array([0, *np.asarray(c, int), dims[a]])
        # the layer after each inner cut copies the one before: >= 2 voxels
        if (np.diff(k) < 2).any():
            raise ValueError(f"cuts on axis {a}: blocks of at least 2 voxels, got {k.tolist()}")
        out.append(k)
    return out


def joint_surface(
    imggrid,
    cuts,
    offset=None,
    simplify_eps=None,
    max_edge_length=np.inf,
    cap_iterations=16,
    **fair_kwargs,
):
    """
    Joint multi-block surface of the label image imggrid (as for
    extract_surface: cell data 'data', 0 = background, made ECS by
    prepare_labels), faired with imagemesh's fair_surface_net (fair_kwargs:
    influence_radius, alpha, dead_zone, max_distance, volume, iterations, ...),
    with caps on the inner cut planes that neighbouring blocks share.

    cuts: blocks per axis (int or 3 ints: equal blocks, rounded to voxels),
    or 3 sequences of the inner cut planes' voxel indices.

    The layer after each inner cut copies the layer before it, so no
    interface lies on a cut plane. The labels of block (i, j, k) are shifted
    by its region index r times offset (default: largest label + 1), so every
    voxel face on a cut plane is a cap between two blocks. The interfaces
    (without the caps, with the unshifted labels, so they are faired as if
    uncut) are faired with the normal coordinate of every point on a cut
    plane fixed; the cap points are then relaxed in their planes
    (cap_iterations Laplacian steps, the interface points fixed), so caps
    stay planar and seam lines straight. The triangulated surface is
    simplified once with simplify_surface (epsilon simplify_eps, default
    0.05 dx, 0 disables; endpoint placement keeps cap points on their planes
    and seam points on their lines).

    Returns the surface with cell data 'boundary_labels' (shifted labels, 0
    outside) and field data 'offset', 'dx' and the cut layout of
    tetopt.blockwise.surfaces.blocks_from_joint_surface: with the same number
    n of blocks on every axis, 'n' and 'cuts' (3 x (n + 1) plane coordinates,
    the outer box included), r = i + n j + n^2 k; otherwise 'x_cuts',
    'y_cuts', 'z_cuts' (T - w, T, T + w per inner plane T, w = dx / pi), whose
    cut intervals are the regions, r = 3 i + 3 m j + 3 m^2 k with m the
    largest number of intervals per axis. 'block_cuts' holds the plane
    coordinates per axis (ragged, flattened) and 'block_n' the blocks per axis.
    """
    from imagemesh.fairing import fair_surface_net
    from imagemesh.simplification import simplify_surface
    from imagemesh.smoothing import build_stencils, constrained_smooth, triangulate_quads
    from imagemesh.surface import surface_quads

    from emimesh.extract_surfaces import prepare_labels

    grid = prepare_labels(imggrid)
    dims = np.array(grid.dimensions) - 1
    spacing = np.array(grid.spacing, np.float64)
    origin = np.array(grid.origin, np.float64)
    dx = float(spacing.min())
    kcut = _cut_indices(cuts, dims)
    nblk = [len(k) - 1 for k in kcut]
    planes = [origin[a] + kcut[a] * spacing[a] for a in range(3)]

    data = np.asarray(grid.cell_data["data"]).reshape(tuple(dims), order="F").astype(np.int64)
    for a in range(3):
        for k in kcut[a][1:-1]:
            dst, src = [slice(None)] * 3, [slice(None)] * 3
            dst[a], src[a] = k, k - 1
            data[tuple(dst)] = data[tuple(src)]
    off = int(data.max()) + 1 if offset is None else int(offset)
    if off <= data.max():
        raise ValueError(f"offset={off} must exceed the largest label {data.max()}")

    cubic = len(set(nblk)) == 1
    if cubic:
        stride = [1, nblk[0], nblk[0] ** 2]
    else:
        m = max(3 * (n - 1) + 1 for n in nblk)
        stride = [3, 3 * m, 3 * m * m]
    reg = np.zeros(tuple(dims), np.int64)
    for a in range(3):
        idx = np.searchsorted(kcut[a][1:-1], np.arange(dims[a]), side="right")
        shape = [1, 1, 1]
        shape[a] = dims[a]
        reg = reg + idx.reshape(shape) * stride[a]
    data = np.where(data > 0, data + reg * off, 0)
    shifted = pv.ImageData(dimensions=grid.dimensions, spacing=grid.spacing, origin=grid.origin)
    shifted.cell_data["data"] = data.ravel(order="F")
    surf = surface_quads(shifted)

    quads = np.asarray(surf.regular_faces, np.int64)
    bl = np.asarray(surf.cell_data["boundary_labels"], np.int64)
    base = np.where(bl > 0, (bl - 1) % off + 1, 0)
    cap = (base[:, 0] == base[:, 1]) & (bl[:, 0] != bl[:, 1])
    P = np.array(surf.points, np.float64)
    # the coordinates on a plane (cut or box face), set exactly to it
    on_plane = np.zeros(P.shape, bool)
    tol = 1e-3 * dx
    for a in range(3):
        for c in planes[a]:
            on = np.abs(P[:, a] - c) <= tol
            P[on, a] = c
            on_plane[on, a] = True

    # 1. fair the interfaces as if uncut: unshifted labels, no caps
    iq = quads[~cap]
    used, inv = np.unique(iq, return_inverse=True)
    sub = pv.PolyData.from_regular_faces(P[used], inv.reshape(-1, 4))
    sub.cell_data["boundary_labels"] = base[~cap]
    faired = fair_surface_net(sub, dx, bounds=grid.bounds, fixed=on_plane[used], **fair_kwargs)
    P[used] = faired.points

    # 2. relax the cap points in their planes, the interface points fixed
    fixed = on_plane.copy()
    fixed[used] = True
    if cap_iterations and cap.any():
        A = build_stencils(quads[cap], len(P), degenerate=None)
        P = constrained_smooth(P, A, iterations=cap_iterations, distance=np.inf, fixed=fixed)
    tris = triangulate_quads(P, quads)
    out = pv.PolyData.from_regular_faces(P, tris)
    out.cell_data["boundary_labels"] = np.repeat(bl, 2, axis=0)

    # 3. simplify once
    eps = 0.05 * dx if simplify_eps is None else simplify_eps
    if eps > 0:
        out = simplify_surface(out, epsilon=eps, max_edge_length=max_edge_length)

    out.field_data["offset"] = [off]
    out.field_data["dx"] = [dx]
    for name in ("influence_radius", "max_surface_distance"):
        out.field_data[name] = faired.field_data[name]
    out.field_data["block_n"] = nblk
    out.field_data["block_cuts"] = np.concatenate(planes)
    if cubic:
        out.field_data["n"] = [nblk[0]]
        out.field_data["cuts"] = np.concatenate(planes)
    else:
        w = dx / np.pi
        for a, c in enumerate("xyz"):
            out.field_data[f"{c}_cuts"] = [x + d for x in planes[a][1:-1] for d in (-w, 0.0, w)]
    return out
