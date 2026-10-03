"""
Replica of the smoothing in vtkSurfaceNets3D (pyvista's contour_labels),
applied to an existing unsmoothed surface net, with optional per-point,
per-axis constraints.

vtkSurfaceNets3D builds one smoothing stencil per point and hands it to
vtkConstrainedSmoothingFilter:

* stencil: the edge neighbours of the point.
* optimized stencils (VTK default): if the point has an edge shared by more
  than two faces (a junction where >= 3 regions meet), only the neighbours
  along junction edges are used, so junction curves are smoothed as curves
  instead of being pulled into the adjacent sheets.
* a stencil with a single neighbour locks the point.
* Jacobi iterations x <- x + relax * (mean(stencil) - x); after each
  iteration the displacement from the original position is clamped to a
  sphere of radius `distance`.
"""

import numpy as np
import pyvista as pv
import scipy.sparse as sp


def unique_faces(faces):
    """Faces with duplicates (same vertex set, any order) removed."""
    _, idx = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return faces[np.sort(idx)]


def build_stencils(faces, n_points, optimized=True):
    """
    Row-normalized sparse averaging matrix A, so that A @ x is the stencil
    mean. faces must be the quads of the unsmoothed surface net; duplicate
    faces are counted once.
    """
    faces = unique_faces(np.asarray(faces))
    edges = np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1).reshape(-1, 2)
    edges, valence = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)

    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    val = np.concatenate([valence, valence])

    if optimized:
        # points touching a junction edge only use junction edges
        junction = np.zeros(n_points, dtype=bool)
        junction[i[val > 2]] = True
        keep = ~junction[i] | (val > 2)
        i, j = i[keep], j[keep]

    # a single neighbour locks the point: its stencil is the point itself
    locked = np.bincount(i, minlength=n_points) == 1
    keep = ~locked[i]
    lid = np.flatnonzero(locked)
    i = np.concatenate([i[keep], lid])
    j = np.concatenate([j[keep], lid])

    n_nb = np.bincount(i, minlength=n_points)
    return sp.csr_matrix((1.0 / n_nb[i], (i, j)), shape=(n_points, n_points))


def constrained_smooth(
    points, A, iterations=16, relaxation=0.5, distance=1.0, convergence=0.0, fixed=None
):
    """
    vtkConstrainedSmoothingFilter with a sphere constraint.

    fixed: optional (n_points, 3) bool mask of coordinates that must not move,
    e.g. fixed[k, 2] = True keeps point k in its z-plane.
    """
    x0 = np.asarray(points, dtype=np.float64)
    x = x0.copy()
    d2 = distance**2
    for _ in range(iterations):
        xn = x + relaxation * (A @ x - x)
        if fixed is not None:
            xn[fixed] = x0[fixed]
        disp = xn - x0
        r2 = np.einsum("ij,ij->i", disp, disp)
        over = r2 > d2
        xn[over] = x0[over] + disp[over] * np.sqrt(d2 / r2[over])[:, None]
        moved = np.abs(xn - x).max()
        x = xn
        if moved <= convergence:
            break
    return x


def bounding_box_mask(points, tol=1e-6):
    """(n_points, 3) mask of the coordinates lying on the bounding box faces."""
    p = np.asarray(points)
    lo, hi = p.min(axis=0), p.max(axis=0)
    atol = tol * np.max(hi - lo)
    return np.isclose(p, lo, rtol=0, atol=atol) | np.isclose(p, hi, rtol=0, atol=atol)


def triangulate_quads(points, quads):
    """
    Split each quad along its shorter diagonal, unless the two triangles would
    fold onto each other (the quad became non-convex during smoothing); then
    the other diagonal is used. Ties are broken by vertex id, so a quad shared
    by two surfaces is split the same way in both, independent of its vertex
    order and orientation.
    """
    quads = np.asarray(quads)
    p = np.asarray(points)
    d02 = np.linalg.norm(p[quads[:, 0]] - p[quads[:, 2]], axis=1)
    d13 = np.linalg.norm(p[quads[:, 1]] - p[quads[:, 3]], axis=1)
    min02 = np.minimum(quads[:, 0], quads[:, 2])
    min13 = np.minimum(quads[:, 1], quads[:, 3])
    use13 = (d13 < d02) | ((d13 == d02) & (min13 < min02))

    def folds(q):
        # normals of the triangles (0, 1, 2) and (0, 2, 3) point in opposite directions
        n1 = np.cross(p[q[:, 1]] - p[q[:, 0]], p[q[:, 2]] - p[q[:, 0]])
        n2 = np.cross(p[q[:, 2]] - p[q[:, 0]], p[q[:, 3]] - p[q[:, 0]])
        return np.einsum("ij,ij->i", n1, n2) < 0

    fold02, fold13 = folds(quads), folds(np.roll(quads, -1, axis=1))
    use13 = np.where(fold02 != fold13, fold02, use13)
    q = np.where(use13[:, None], np.roll(quads, -1, axis=1), quads)
    return np.stack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=1).reshape(-1, 3)


def smooth_surface_net(
    surf,
    iterations=16,
    relaxation=0.5,
    distance=None,
    scale=1.0,
    spacing=None,
    optimized=True,
    fixed=None,
    fix_bounds=False,
    triangulate=True,
):
    """
    Smooth a surface from contour_labels(smoothing=False, output_mesh_type="quads")
    like contour_labels(smoothing=True) does. Defaults match pyvista:
    distance = norm(spacing) * scale.

    fixed: (n_points, 3) bool mask of coordinates that must not move.
    fix_bounds: additionally keep points on the bounding box on their box face.
    """
    quads = surf.regular_faces
    if distance is None:
        distance = np.linalg.norm(spacing)
    A = build_stencils(quads, surf.n_points, optimized=optimized)

    fixed = np.zeros((surf.n_points, 3), dtype=bool) if fixed is None else np.asarray(fixed)
    if fix_bounds:
        fixed = fixed | bounding_box_mask(surf.points)

    pts = constrained_smooth(surf.points, A, iterations, relaxation, distance * scale, fixed=fixed)
    if not triangulate:
        out = surf.copy()
        out.points = pts
        return out
    tris = triangulate_quads(pts, quads)
    out = pv.PolyData(pts, np.column_stack([np.full(len(tris), 3), tris]).ravel())
    for name, arr in surf.cell_data.items():
        out.cell_data[name] = np.repeat(arr, 2, axis=0)
    return out
