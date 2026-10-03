"""
Stitched mid block input for dt, with explicit per-cell stitching tubes: the mid surface
(with its own labelled caps, compressed affinely in z into [z_lo + d, z_hi - d],
which keeps it closed and intersection free), the tetwild
cut faces at z_lo/z_hi, and in each slab between the two caps:

* for every cell (label != ECS) and every boundary cycle of it on the tetwild cap, a tube
  zipping it to the matching boundary cycle of the same cell on the mid cap (only vertices
  on the membrane of that cell are connected; the zip minimizes the total rung length).
  Cycles touching the box side are split into interior chains (rim to rim), which are
  zipped separately; their end points become anchors of the side wall strips.
* side wall strips between the two cap rims, zipped piecewise between the anchors.

usage: build_mid.py [d] [surface]
"""

import sys

import numba as nb
import numpy as np
import pyvista as pv
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from vtkio import write_surface

d = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
fname = sys.argv[2] if len(sys.argv) > 2 else "mid_dec.vtk"
z_lo, z_hi = 950.0, 1050.0
ECS = 1
x0, x1, y0, y1 = 0.0, 2000.0, 0.0, 2000.0


def rim_loop(tris):
    """ordered vertex loop of the boundary of a disk-like triangulation"""
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    b = u[c == 1]
    nb = {}
    for i, j in b:
        nb.setdefault(i, []).append(j)
        nb.setdefault(j, []).append(i)
    assert all(len(v) == 2 for v in nb.values()), "rim is not a simple loop"
    loop = [b[0, 0], b[0, 1]]
    while True:
        a, c_ = nb[loop[-1]]
        n = a if a != loop[-2] else c_
        if n == loop[0]:
            break
        loop.append(n)
    assert len(loop) == len(nb), f"rim has several loops ({len(loop)} of {len(nb)} vertices)"
    return np.array(loop)


def orient_and_split(loop, P):
    """counterclockwise loop in xy, split into the 4 box sides at the vertices nearest the corners"""
    xy = P[loop, :2]
    if np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]) < 0:
        loop = loop[::-1]
        xy = xy[::-1]
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    ci = [np.argmin(np.linalg.norm(xy - c, axis=1)) for c in corners]
    loop = np.roll(loop, -ci[0])
    ci = [(c - ci[0]) % len(loop) for c in ci] + [len(loop)]
    assert ci == sorted(ci), "corners out of order"
    ext = np.append(loop, loop[0])
    return [ext[ci[k] : ci[k + 1] + 1] for k in range(4)]


def zip_polylines(a, b, P):
    """
    Triangle strip between two polylines with matching end points, advancing by the
    position along the side (no crossings when both are monotone along it).
    """
    t = P[a[-1]] - P[a[0]]
    t[2] = 0
    sa, sb = P[a] @ t, P[b] @ t
    tris, i, j = [], 0, 0
    while i < len(a) - 1 or j < len(b) - 1:
        if j == len(b) - 1 or (i < len(a) - 1 and sa[i + 1] <= sb[j + 1]):
            tris.append((a[i], a[i + 1], b[j]))
            i += 1
        else:
            tris.append((a[i], b[j + 1], b[j]))
            j += 1
    return tris


def ccw(P, tris):
    """triangles oriented counterclockwise in xy"""
    a = P[tris[:, 1], :2] - P[tris[:, 0], :2]
    b = P[tris[:, 2], :2] - P[tris[:, 0], :2]
    t = tris.copy()
    flip = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] < 0
    t[flip] = t[flip][:, [0, 2, 1]]
    return t


def region_cycles(tris, lab, L):
    """
    Boundary cycles of region L (tris ccw, so the region is on the left). At a pinch
    vertex the boundary continues around the same fan; cycles touching at a pinch vertex
    are then concatenated into one cycle (non crossing, both lobes ccw).
    """
    third = {}
    for a, b, c in tris[lab == L]:
        third[(a, b)], third[(b, c)], third[(c, a)] = c, a, b
    bnd = [e for e in third if (e[1], e[0]) not in third]
    nxt = {}
    for u, v in bnd:
        w = third[(u, v)]
        while (w, v) in third:
            w = third[(w, v)]
        nxt[(u, v)] = (v, w)
    loops, seen = [], set()
    for e0 in bnd:
        if e0 in seen:
            continue
        loop, e = [], e0
        while e not in seen:
            seen.add(e)
            loop.append(e[0])
            e = nxt[e]
        loops.append(loop)
    merged = True
    while merged:
        merged = False
        for i in range(len(loops)):
            for j in range(i + 1, len(loops)):
                common = set(loops[i]) & set(loops[j])
                if common:
                    v = min(common)
                    a, b = loops[i], loops[j]
                    ia, ib = a.index(v), b.index(v)
                    loops[i] = a[ia:] + a[:ia] + b[ib:] + b[:ib]
                    del loops[j]
                    merged = True
                    break
            if merged:
                break
    return [np.array(lp) for lp in loops]


@nb.njit(cache=True)
def _zip_dp(D):
    n, m = D.shape
    C = np.full((n, m), np.inf)
    C[0, 0] = 0.0
    for i in range(n):
        for j in range(m):
            if i > 0 and C[i - 1, j] + D[i, j] < C[i, j]:
                C[i, j] = C[i - 1, j] + D[i, j]
            if j > 0 and C[i, j - 1] + D[i, j] < C[i, j]:
                C[i, j] = C[i, j - 1] + D[i, j]
    # backtrack: 0 = advance a, 1 = advance b
    steps = np.empty(n + m - 2, np.int64)
    i, j, k = n - 1, m - 1, n + m - 3
    while k >= 0:
        if j == 0 or (i > 0 and C[i - 1, j] <= C[i, j - 1]):
            steps[k] = 0
            i -= 1
        else:
            steps[k] = 1
            j -= 1
        k -= 1
    return steps


def zip_dp(a, b, P):
    """strip between polylines a and b (a[0]-b[0] and a[-1]-b[-1] are rungs), min total rung length"""
    D = np.linalg.norm(P[a][:, None] - P[b][None], axis=2)
    tris, i, j = [], 0, 0
    for s in _zip_dp(D):
        if s == 0:
            tris.append((a[i], a[i + 1], b[j]))
            i += 1
        else:
            tris.append((a[i], b[j + 1], b[j]))
            j += 1
    return tris


def zip_closed(A, B, P):
    j0 = np.argmin(np.linalg.norm(P[B] - P[A[0]], axis=1))
    B = np.roll(B, -j0)
    return zip_dp(np.append(A, A[0]), np.append(B, B[0]), P)


def interior_chains(c, rim):
    """split a cycle at its rim edges into interior chains (None if it does not touch the rim)"""
    n = len(c)
    r = np.array([frozenset((c[k], c[(k + 1) % n])) in rim for k in range(n)])
    if not r.any():
        return None
    k0 = np.flatnonzero(r)[0]
    c, r = np.roll(c, -(k0 + 1)), np.roll(r, -(k0 + 1))
    chains, cur = [], [c[0]]
    for k in range(n):
        if r[k]:
            if len(cur) > 1:
                chains.append(np.array(cur))
            cur = [c[(k + 1) % n]]
        else:
            cur.append(c[(k + 1) % n])
    if len(cur) > 1:
        chains.append(np.array(cur))
    return chains


def signed_area(c, P):
    x, y = P[c, 0], P[c, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def loop_dist(a, b, P):
    return 0.5 * (cKDTree(P[b]).query(P[a])[0].mean() + cKDTree(P[a]).query(P[b])[0].mean())


def collar(A, lift, closed):
    """
    vertical strip from the tetwild polyline A up to its copy A' on the collar plane (rim
    vertices stay, so nothing lies in the side walls); the tetwild cap is a height field,
    so the strip cannot cut it, and the tube from A' to the flat mid cap stays above it
    """
    A2 = np.array([lift.get(v, v) for v in A])
    tris = []
    for k in range(len(A) if closed else len(A) - 1):
        a, b, a2, b2 = A[k], A[(k + 1) % len(A)], A2[k], A2[(k + 1) % len(A)]
        if b2 != b:
            tris.append((a, b, b2))
        if a2 != a:
            tris.append((a, b2, a2))
    return A2, tris


def stitch(P, Tc, lc, Tm, lm, side, lift):
    """tubes between the tetwild cap (Tc, labels lc) and the mid cap (Tm, labels lm)"""
    Tc, Tm = ccw(P, Tc), ccw(P, Tm)
    rim_c, rim_m = rim_loop(Tc), rim_loop(Tm)
    rim_set = {frozenset(e) for r in [rim_c, rim_m] for e in zip(r, np.roll(r, -1))}
    tubes, tube_lab, anchors, report = [], [], {}, []
    for L in sorted(set(lc) | set(lm)):
        if L == ECS:
            continue
        cc = region_cycles(Tc, lc, L) if (lc == L).any() else []
        cm = region_cycles(Tm, lm, L) if (lm == L).any() else []
        if not cc or not cm:
            report.append(f"label {L}: only on the {'mid' if cm else 'tetwild'} side, ends flat")
            continue
        cost = np.array(
            [
                [
                    loop_dist(a, b, P)
                    + 1e6 * (np.sign(signed_area(a, P)) != np.sign(signed_area(b, P)))
                    for b in cm
                ]
                for a in cc
            ]
        )
        ia, ib = linear_sum_assignment(cost)
        if len(cc) != len(cm):
            report.append(f"label {L}: {len(cc)} tetwild vs {len(cm)} mid cycles")
        for i, j in zip(ia, ib):
            A, B = cc[i], cm[j]
            if cost[i, j] > 1e5:
                report.append(f"label {L}: cycle orientation mismatch, skipped")
                continue
            ca, cb = interior_chains(A, rim_set), interior_chains(B, rim_set)
            if ca is None or cb is None:
                A2, t = collar(A, lift, True)
                t += zip_closed(A2, B, P)
            elif len(ca) == len(cb):
                cst = np.array(
                    [
                        [
                            np.linalg.norm(P[x[0]] - P[y[0]]) + np.linalg.norm(P[x[-1]] - P[y[-1]])
                            for y in cb
                        ]
                        for x in ca
                    ]
                )
                t = []
                for p, q in zip(*linear_sum_assignment(cst)):
                    A2, tc = collar(ca[p], lift, False)
                    t += tc + zip_dp(A2, cb[q], P)
                    anchors[ca[p][0]], anchors[ca[p][-1]] = cb[q][0], cb[q][-1]
            elif min(len(ca), len(cb)) == 1:
                # one chain against several: join the several, with the rim pieces between
                # them, into one chain spanning from the end points nearest the single one
                one, many, cyc = (ca[0], cb, B) if len(ca) == 1 else (cb[0], ca, A)
                n = len(cyc)
                pos = {v: k for k, v in enumerate(cyc)}
                best = None
                for g in range(len(many)):
                    s0, e0 = many[(g + 1) % len(many)][0], many[g][-1]
                    sp = cyc[(pos[s0] + np.arange((pos[e0] - pos[s0]) % n + 1)) % n]
                    c_ = np.linalg.norm(P[sp[0]] - P[one[0]]) + np.linalg.norm(
                        P[sp[-1]] - P[one[-1]]
                    )
                    if best is None or c_ < best[0]:
                        best = (c_, sp)
                x, y = (one, best[1]) if len(ca) == 1 else (best[1], one)
                x2, t = collar(x, lift, False)
                t += zip_dp(x2, y, P)
                anchors[x[0]], anchors[x[-1]] = y[0], y[-1]
                report.append(
                    f"label {L}: {len(ca)} vs {len(cb)} interior chains at the rim, joined"
                )
            else:
                report.append(
                    f"label {L}: {len(ca)} vs {len(cb)} interior chains at the rim, skipped"
                )
                continue
            tubes += t
            tube_lab += [L] * len(t)
    # side wall strips, split at the anchors
    walls = []
    for a, b in zip(orient_and_split(rim_c, P), orient_and_split(rim_m, P)):
        pa = {v: k for k, v in enumerate(a)}
        pb = {v: k for k, v in enumerate(b)}
        cut = sorted(
            (pa[u], pb[v]) for u, v in anchors.items() if u in pa and 0 < pa[u] < len(a) - 1
        )
        cut = [(0, 0)] + cut + [(len(a) - 1, len(b) - 1)]
        assert all(cut[k][1] <= cut[k + 1][1] for k in range(len(cut) - 1)), "anchors cross"
        for (i0, j0), (i1, j1) in zip(cut[:-1], cut[1:]):
            walls += zip_polylines(a[i0 : i1 + 1], b[j0 : j1 + 1], P)
    print(f"{side}: tubes {len(tubes)} tris, walls {len(walls)} tris, anchors {len(anchors)}")
    for r in report:
        print("   ", r)
    return np.array(tubes), np.array(tube_lab), np.array(walls)


top, bot = pv.read("top_cut.vtk"), pv.read("bottom_cut.vtk")
S = pv.read(fname).clean()
q, F, bl = S.points.astype(np.float64), S.regular_faces, S.cell_data["boundary_labels"]
q[:, 2] = z_lo + d + (q[:, 2] - z_lo) * (z_hi - z_lo - 2 * d) / (z_hi - z_lo)
on_lo, on_hi = np.isclose(q[:, 2], z_lo + d), np.isclose(q[:, 2], z_hi - d)
q[on_lo, 2], q[on_hi, 2] = z_lo + d, z_hi - d
cap_lo, cap_hi = on_lo[F].all(axis=1), on_hi[F].all(axis=1)

P = np.vstack([top.points, bot.points, q]).astype(np.float64)
n_cap = top.n_points + bot.n_points
T_top, T_bot, T_S = top.regular_faces, bot.regular_faces + top.n_points, F + n_cap
lab_S = bl.max(axis=1)


def lifted(C, offset, z0, sgn):
    """collar plane copies of the label boundary vertices of a tetwild cap (rim excluded)"""
    t, lab = C.regular_faces, C.cell_data["marker"]
    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    f = np.tile(lab, 3)
    es = np.sort(e, axis=1)
    o = np.lexsort(es.T[::-1])
    es, f = es[o], f[o]
    same = np.all(es[1:] == es[:-1], axis=1)
    v = np.unique(es[:-1][same & (f[1:] != f[:-1])])
    v = np.setdiff1d(v, rim_loop(t))
    zc = z0 + sgn * 0.5 * (np.abs(C.points[:, 2] - z0).max() + d)
    X = C.points[v].astype(np.float64)
    X[:, 2] = zc
    return v + offset, X, zc


v_lo, X_lo, zc_lo = lifted(top, 0, z_lo, 1)
v_hi, X_hi, zc_hi = lifted(bot, top.n_points, z_hi, -1)
n0 = len(P)
P = np.vstack([P, X_lo, X_hi])
lift_lo = dict(zip(v_lo, n0 + np.arange(len(v_lo))))
lift_hi = dict(zip(v_hi, n0 + len(v_lo) + np.arange(len(v_hi))))
print(f"collar planes z={zc_lo:.2f}, {zc_hi:.2f}; lifted vertices {len(v_lo)} + {len(v_hi)}")

tube_lo, tl_lo, wall_lo = stitch(
    P, T_top, top.cell_data["marker"], T_S[cap_lo], lab_S[cap_lo], f"z={z_lo}", lift_lo
)
tube_hi, tl_hi, wall_hi = stitch(
    P, T_bot, bot.cell_data["marker"], T_S[cap_hi], lab_S[cap_hi], f"z={z_hi}", lift_hi
)
T_wall, T_tube = np.vstack([wall_lo, wall_hi]), np.vstack([tube_lo, tube_hi])
T = np.vstack([T_top, T_bot, T_wall, T_tube, T_S])
np.save("mid_in_pts.npy", P)
np.save("mid_in_faces.npy", T)
np.savez(
    "mid_in_meta.npz",
    ntop=len(T_top),
    nbot=len(T_bot),
    nwall=len(T_wall),
    ntube=len(T_tube),
    n_cap=n_cap,
    bl=bl,
    cap_lo=cap_lo,
    cap_hi=cap_hi,
    d=d,
    tube_lab=np.concatenate([tl_lo, tl_hi]),
)
write_surface("mid_in.vtk", P, T)
st = pv.PolyData.from_regular_faces(P, T_tube)
st.cell_data["label"] = np.concatenate([tl_lo, tl_hi])
st.save("mid_stitch.vtk")
