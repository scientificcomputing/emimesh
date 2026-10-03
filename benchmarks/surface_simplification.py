"""
Benchmark imagemesh.simplification against pyvista's decimate
(vtkQuadricDecimation) at the same number of output faces.

usage: python benchmarks/surface_simplification.py surf.vtk [--eps 0.25 0.5 1]
       [--h 10] [--placement endpoint qem] [--tetwild]
"""

import argparse
import time

import igl
import numpy as np
import pyvista as pv

from imagemesh.simplification import simplify_surface


def edge_valence_hist(F):
    e = np.sort(np.stack([F, np.roll(F, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
    _, count = np.unique(e, axis=0, return_counts=True)
    return np.bincount(count, minlength=4)


def label_surface(F, labels, c):
    """Faces bounding label c, oriented like in mark_mesh."""
    return np.vstack([F[labels[:, 0] == c], F[labels[:, 1] == c][:, [0, 2, 1]]])


def open_edges(F, labels):
    """
    Number of edges where a label's oriented surface is not closed (sum over
    labels, label 0 excluded). 0 means each label is a closed, consistently
    oriented surface, as the winding number marking needs.
    """
    n = F.max() + 1
    total = 0
    for c in np.unique(labels):
        if c == 0:
            continue
        Fc = label_surface(F, labels, c)
        a = Fc.ravel()
        b = np.roll(Fc, -1, axis=1).ravel()
        key = np.minimum(a, b) * n + np.maximum(a, b)
        ukey, inv = np.unique(key, return_inverse=True)
        s = np.zeros(len(ukey), np.int64)
        np.add.at(s, inv, np.where(a < b, 1, -1))
        total += np.count_nonzero(s)
    return total


def oriented_normals(V, F, labels):
    """Unit normals, flipped where the label pair is stored in decreasing order."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    return np.where((labels[:, 0] > labels[:, 1])[:, None], -n, n)


def orientation_errors(Vi, Fi, Li, Vo, Fo, Lo, tol):
    """
    Fraction of output faces that are flipped against the input: there is no
    input face with the same (unordered) label pair and a label-oriented
    normal less than 90 degrees apart within tol of the face centroid. Also
    the fraction of output faces whose label pair does not occur in the input.
    """
    Ni, No = oriented_normals(Vi, Fi, Li), oriented_normals(Vo, Fo, Lo)
    pi, po = np.sort(Li, axis=1), np.sort(Lo, axis=1)
    C = Vo[Fo].mean(axis=1)
    flipped = np.zeros(len(Fo), bool)
    missing = np.ones(len(Fo), bool)
    for pair in np.unique(po, axis=0):
        o = np.flatnonzero((po == pair).all(axis=1))
        i = np.flatnonzero((pi == pair).all(axis=1))
        if len(i) == 0:
            continue
        missing[o] = False
        # only the faces whose closest input face is flipped need a closer look
        _, idx, _ = igl.point_mesh_squared_distance(C[o], Vi, Fi[i])
        for f in o[np.einsum("ij,ij->i", No[o], Ni[i[idx]]) <= 0]:
            same = i[Ni[i] @ No[f] > 0]
            d2 = igl.point_mesh_squared_distance(C[f : f + 1], Vi, Fi[same])[0][0]
            flipped[f] = len(same) == 0 or d2 > tol**2
    return flipped.mean(), missing.mean()


def winding_labels(V, F, labels, Q):
    marker = np.zeros(len(Q), np.int64)
    for c in np.unique(labels):
        if c == 0:
            continue
        w = igl.fast_winding_number(V, label_surface(F, labels, c), Q)
        marker = np.where((marker == 0) & (np.abs(w) > 0.5), c, marker)
    return marker


def sample_surface(V, F, n, rng):
    """Area weighted random points on the faces, plus all face centroids."""
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    f = rng.choice(len(F), size=n, p=area / area.sum())
    r = rng.random((n, 2))
    flip = r.sum(axis=1) > 1
    r[flip] = 1 - r[flip]
    T = V[F[f]]
    P = T[:, 0] + r[:, :1] * (T[:, 1] - T[:, 0]) + r[:, 1:] * (T[:, 2] - T[:, 0])
    return np.vstack([P, V[F].mean(axis=1), V])


def triangle_quality(V, F):
    """4 sqrt(3) area / sum(l^2) and the minimal angle (degrees) per face."""
    e = [V[F[:, (i + 1) % 3]] - V[F[:, i]] for i in range(3)]
    l2 = np.stack([np.einsum("ij,ij->i", x, x) for x in e], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(e[0], -e[2]), axis=1)
    q = 4 * np.sqrt(3) * area / l2.sum(axis=1)
    cos = []
    for i in range(3):
        a, b = e[i], -e[(i + 2) % 3]
        cos.append(np.einsum("ij,ij->i", a, b) / np.sqrt(l2[:, i] * l2[:, (i + 2) % 3]))
    min_angle = np.degrees(np.arccos(np.clip(np.max(cos, axis=0), -1, 1)))
    return q, min_angle


def evaluate(name, surf_in, surf_out, h, eps, runtime, query, labels_in_q, rng):
    Vi, Fi = np.asarray(surf_in.points), surf_in.regular_faces
    Vo, Fo = np.asarray(surf_out.points), surf_out.regular_faces
    Lo = surf_out.cell_data["boundary_labels"]

    # two sided deviation, sampled
    Pi = sample_surface(Vi, Fi, 2 * len(Fi), rng)
    Po = sample_surface(Vo, Fo, 2 * len(Fi), rng)
    d_io = np.sqrt(igl.point_mesh_squared_distance(Pi, Vo, Fo)[0]) / h
    d_oi = np.sqrt(igl.point_mesh_squared_distance(Po, Vi, Fi)[0]) / h
    flipped, pair_missing = orientation_errors(
        Vi, Fi, surf_in.cell_data["boundary_labels"], Vo, Fo, Lo, max(eps, 1e-6) * h
    )

    q, min_angle = triangle_quality(Vo, Fo)
    val = edge_valence_hist(Fo)

    # winding number labels of random points, as in mark_mesh
    labels_out_q = winding_labels(Vo, Fo, Lo, query)
    dq = np.sqrt(igl.point_mesh_squared_distance(query, Vi, Fi)[0])
    wrong = labels_out_q != labels_in_q
    far = dq > 2 * eps * h if eps else dq > 2 * h

    return dict(
        method=name,
        faces=len(Fo),
        time_s=runtime,
        in_to_out_max=d_io.max(),
        in_to_out_p99=np.percentile(d_io, 99),
        in_to_out_mean=d_io.mean(),
        out_to_in_max=d_oi.max(),
        out_to_in_p99=np.percentile(d_oi, 99),
        out_to_in_mean=d_oi.mean(),
        q_min=q.min(),
        q_below_0_1=np.mean(q < 0.1),
        q_mean=q.mean(),
        min_angle_min=min_angle.min(),
        min_angle_p1=np.percentile(min_angle, 1),
        valence1=val[1],
        valence_gt2=val[3:].sum(),
        open_label_edges=open_edges(Fo, Lo),
        flipped=flipped,
        pair_missing=pair_missing,
        wn_mismatch=wrong.mean(),
        wn_mismatch_far=wrong[far].mean(),
    )


def transfer_labels(surf_in, surf_out):
    """Labels of the nearest input face to each output face centroid."""
    C = np.asarray(surf_out.cell_centers().points)
    _, idx, _ = igl.point_mesh_squared_distance(
        C, np.asarray(surf_in.points), surf_in.regular_faces
    )
    out = surf_out.copy()
    out.cell_data["boundary_labels"] = surf_in.cell_data["boundary_labels"][idx]
    return out


def run_tetwild(surf, simplify):
    import pytetwild

    t = time.perf_counter()
    mesh = pytetwild.tetrahedralize_pv(
        surf,
        stop_energy=10,
        quiet=True,
        disable_filtering=True,
        edge_length_fac=0.05,
        epsilon=1e-3,
        coarsen=False,
        num_threads=6,
        simplify=simplify,
    )
    return time.perf_counter() - t, mesh.n_cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("surf")
    parser.add_argument("--eps", type=float, nargs="+", default=[0.25, 0.5, 1.0], help="in voxels")
    parser.add_argument("--h", type=float, default=10.0, help="voxel size")
    parser.add_argument("--nquery", type=int, default=200_000)
    parser.add_argument(
        "--placement", nargs="+", default=["endpoint", "qem"], choices=["endpoint", "qem"]
    )
    parser.add_argument("--tetwild", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    surf = pv.read(args.surf)
    surf.cell_data["boundary_labels"] = np.asarray(surf.cell_data["boundary_labels"])
    V, F = np.asarray(surf.points), surf.regular_faces
    lo, hi = V.min(axis=0), V.max(axis=0)
    query = lo + rng.random((args.nquery, 3)) * (hi - lo)
    labels_in_q = winding_labels(V, F, surf.cell_data["boundary_labels"], query)
    print(f"{args.surf}: {surf.n_points} points, {surf.n_cells} faces")

    # compile the numba kernels
    small = surf.extract_cells(range(1000)).extract_surface(algorithm="dataset_surface")
    for placement in args.placement:
        simplify_surface(small, args.eps[0], placement=placement)

    rows = [evaluate("input", surf, surf, args.h, 0, 0.0, query, labels_in_q, rng)]
    tetwild = {}
    if args.tetwild:
        tetwild["input"] = run_tetwild(surf, simplify=True)
    for eps in args.eps:
        runs = []
        for placement in args.placement:
            t = time.perf_counter()
            ours = simplify_surface(surf, eps * args.h, placement=placement)
            runs.append((f"{placement} eps={eps}", ours, time.perf_counter() - t))
        # decimate to the size of the first of ours
        t = time.perf_counter()
        dec = surf.decimate(1 - runs[0][1].n_cells / surf.n_cells)
        t_dec = time.perf_counter() - t
        runs.append((f"decimate @{eps}", transfer_labels(surf, dec), t_dec))
        for name, s, rt in runs:
            rows.append(evaluate(name, surf, s, args.h, eps, rt, query, labels_in_q, rng))
            if args.tetwild:
                tetwild[name] = run_tetwild(s, simplify=False)
            if args.save:
                s.save(f"{name.replace(' ', '_').replace('=', '').replace('@', '')}.vtk")

    keys = list(rows[0])
    print("\t".join(keys))
    for r in rows:
        print("\t".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in keys))
    if tetwild:
        print("\nfTetWild (simplify only for the input): time [s], tets")
        for k, (t, n) in tetwild.items():
            print(f"{k}\t{t:.1f}\t{n}")


if __name__ == "__main__":
    main()
