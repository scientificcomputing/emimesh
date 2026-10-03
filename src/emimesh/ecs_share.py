"""
Prescribe the extracellular space (ECS) volume share of a multi-label surface
(e.g. top_surf_smooth.vtk from split_blocks.py) by moving all cell-ECS
interfaces by the same normal distance d, found by root finding on the share.

Conventions: boundary_labels (a, b) per triangle, the normal points out of
region a into region b; 0 is outside the box, 1 the ECS, >= 2 the cells.
Every cell is wrapped in ECS (there are no cell-cell faces) and capped by
faces (cell, 0) / (ECS, 0) on the box faces.

* Offset: d > 0 shrinks the cells (more ECS), d < 0 grows them. The offset is
  applied as a flow in small steps: the normals are recomputed after each
  step, each step's displacement and then the positions are Taubin smoothed
  along the interface, so fronts that converge round off instead of forming
  creases or spikes.
* Box: points on a box face move only within it, and the normal component
  towards a box face fades out over `blend` from it, so the outer dimensions
  never change. The caps follow the interfaces by a harmonic extension of the
  in-plane displacement.
* Limits: per point, the local thickness of the cell and of the ECS gap (the
  diameter of the largest inscribed ball reaching the point, computed on the
  surface) limits the offset so that both keep at least `min_width`; the
  limits are slope limited along the surface, so the offset varies smoothly.
* Validity: after each solve, flipped and self-intersecting triangles are
  detected, the limits around them are reduced and d is solved again.
"""

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from imagemesh.smoothing import bounding_box_mask
from imagemesh.surface import flipped_faces, label_volumes, self_intersections

OUT, ECS = 0, 1

# distances along the normal at which inscribed ball centres are placed
BALL_STEPS = np.r_[
    np.arange(0.5, 5.01, 0.5), np.arange(6, 12.01, 1), np.arange(14, 20.01, 2), 25, 30, 35, 40
]


def ecs_share(V, F, L):
    """ECS volume / volume of the box."""
    vol = label_volumes(V, F, L)
    return vol[ECS] / vol[1:].sum()


def cell_normals(V, F, L, mask):
    """Area weighted unit vertex normals of the faces in mask (cell-ECS),
    pointing out of the cells; ok marks the points with a normal."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    fn[L[:, 0] == ECS] *= -1
    fn[~mask] = 0
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 0
    n[ok] /= ln[ok, None]
    return n, ok


def edges(F):
    """Unique undirected edges of the faces."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def umbrella(F, n_points):
    """Row normalised adjacency matrix of the faces' edges, and the mask of
    the points that have edges."""
    e = edges(F)
    e = np.concatenate([e, e[:, ::-1]])
    A = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_points, n_points))
    deg = np.asarray(A.sum(1)).ravel()
    return sp.diags(1 / np.maximum(deg, 1)) @ A, deg > 0


def taubin(u, S, iters, keep, lam=0.5, mu=-0.53):
    """Taubin lambda/mu smoothing of the field u with the umbrella operator S
    (no shrinkage of the large scales); rows in keep are held."""
    A, has = S
    free = (has & ~keep)[:, None]
    for _ in range(iters):
        for f in (lam, mu):
            u = np.where(free, u + f * (A @ u - u), u)
    return u


def lipschitz(c, V, F, slope):
    """Largest field <= c with |c_i - c_j| <= slope |x_i - x_j| along the
    edges of F (min-plus relaxation)."""
    e = edges(F)
    e = np.concatenate([e, e[:, ::-1]])
    w = slope * np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    c = c.copy()
    while True:
        new = c.copy()
        np.minimum.at(new, e[:, 0], c[e[:, 1]] + w)
        if np.array_equal(new, c):
            return c
        c = new


def local_thickness(V, n, idx, iface, side, steps=BALL_STEPS, dr=0.5, tol=1.0):
    """
    Local thickness at the points V[idx] on one side of the interface surface
    iface (oriented out of the cells; side = -1: into the cell, +1: into the
    ECS). Ball centres are placed at V[idx] + side * s * n for s in steps; a
    centre's radius is its distance to iface, and centres that ended up on the
    other side are dropped. A point gets 2 * the largest radius (binned to dr,
    rounded down) of a ball that reaches it within tol. The box faces are not
    walls.
    """
    p, nn = V[idx], n[idx]
    C = (p[None] + side * steps[:, None, None] * nn[None]).reshape(-1, 3)
    D = pv.PointSet(C).compute_implicit_distance(iface)["implicit_distance"]
    keep = np.sign(D) == side
    C, R = C[keep], np.abs(D[keep])
    band = np.floor(R / dr).astype(int)
    t = np.zeros(len(idx))
    for b in np.unique(band):
        if b == 0:
            continue
        r = b * dr
        dist, _ = cKDTree(C[band == b]).query(p, distance_upper_bound=r + tol)
        t = np.where(dist <= r + tol, np.maximum(t, 2 * r), t)
    return t


def coincident_partners(V, moving, tol=1e-2):
    """For points within tol of a moving point, the index of that point, else
    -1. Such pairs occur on the cut planes of split_blocks.py, where the
    smoothing collapses two vertices of a quad onto each other."""
    mi = np.flatnonzero(moving)
    dist, j = cKDTree(V[mi]).query(V, distance_upper_bound=tol)
    partner = np.where(np.isfinite(dist), mi[np.minimum(j, len(mi) - 1)], -1)
    partner[moving] = -1
    return partner


def extend_caps(V0, F, L, disp, moving, partner, pinned=None):
    """
    Harmonic extension of the in-plane displacement into the box faces (faces
    touching label 0): cap points that are not moved by the offset and lie on
    exactly one box plane are relaxed; points coincident with a moving point
    follow it; box edge/corner points and pinned points are kept.
    """
    disp = disp.copy()
    follow = partner >= 0
    disp[follow] = disp[partner[follow]]
    Fb = F[(L == OUT).any(1)]
    on_cap = np.zeros(len(V0), bool)
    on_cap[Fb.ravel()] = True
    free = on_cap & ~moving & ~follow & (bounding_box_mask(V0).sum(1) == 1)
    if pinned is not None:
        free &= ~pinned
    e = edges(Fb)
    e = np.concatenate([e, e[:, ::-1]])
    n = len(V0)
    W = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    Lap = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    fi = np.flatnonzero(free)
    ci = np.flatnonzero(on_cap & ~free)
    solve = spla.factorized(Lap[fi][:, fi].tocsc())
    Lfc = Lap[fi][:, ci]
    for k in range(3):
        disp[fi, k] = solve(-(Lfc @ disp[ci, k]))
    disp[bounding_box_mask(V0)] = 0
    return disp


def one_ring(mask, F):
    """mask grown by the vertices of the faces touching it."""
    m = mask.copy()
    m[F[mask[F].any(1)].ravel()] = True
    return m


class ECSAdjuster:
    """
    Offset field of the cell-ECS interfaces of a multi-label surface (see the
    module docstring). points(d) are the interface points after the offset d,
    share(d) the resulting ECS share, solve(target) finds d.
    """

    def __init__(
        self,
        surf,
        min_width=10.0,
        slope=0.25,
        blend=20.0,
        steps=8,
        smooth_iters=5,
        pos_iters=2,
        verbose=True,
    ):
        self.F = surf.regular_faces
        self.L = surf.cell_data["boundary_labels"].astype(int)
        self.V0 = surf.points.astype(np.float64)
        F, L, V = self.F, self.L, self.V0
        self.blend, self.steps = blend, steps
        self.smooth_iters, self.pos_iters = smooth_iters, pos_iters
        self.box_lo, self.box_hi = V.min(0), V.max(0)
        self.fixed = bounding_box_mask(V)
        self.cell_ecs = (L.min(1) == ECS) & (L.max(1) > ECS)

        # points that move: on the interface, not mostly towards a box face
        n_raw, ok = cell_normals(V, F, L, self.cell_ecs)
        n = n_raw * self.fade(V)
        n[self.fixed] = 0
        self.moving = ok & (np.linalg.norm(n, axis=1) > 0.2)
        self.partner = coincident_partners(V, self.moving)
        self.held = np.zeros(len(V), bool)
        self.S = umbrella(F[self.cell_ecs], len(V))
        self.Fi = F[self.cell_ecs]

        # local thickness of the cells and of the ECS gaps
        Fi = F[self.cell_ecs].copy()
        flip = L[self.cell_ecs, 0] == ECS
        Fi[flip] = Fi[flip][:, ::-1]
        iface = pv.PolyData.from_regular_faces(V, Fi)
        idx = np.flatnonzero(self.moving)
        self.t_in = np.zeros(len(V))
        self.t_out = np.zeros(len(V))
        self.t_in[idx] = local_thickness(V, n_raw, idx, iface, -1)
        self.t_out[idx] = local_thickness(V, n_raw, idx, iface, +1)
        # shrinking a cell by d thins it by 2 d; growing the cells on both
        # sides of an ECS gap by d narrows it by 2 d
        self.slope = slope
        m = self.moving
        self.max_shrink = self.limit(np.where(m, np.maximum(0.5 * (self.t_in - min_width), 0), 0))
        self.max_grow = self.limit(np.where(m, np.maximum(0.5 * (self.t_out - min_width), 0), 0))
        if verbose:
            print(
                f"ECSAdjuster: {m.sum()} moving points, input ECS share {ecs_share(V, F, L):.4f}",
                flush=True,
            )

    def fade(self, V):
        """Factors of the normal components: 0 on a box face, 1 beyond blend."""
        return np.clip(np.minimum(V - self.box_lo, self.box_hi - V) / self.blend, 0, 1)

    def limit(self, c):
        return lipschitz(c, self.V0, self.Fi, self.slope)

    def offsets(self, d):
        """Per point offset (> 0 shrinks the cells) for the global offset d."""
        return np.clip(d, -self.max_grow, self.max_shrink) * self.moving

    def normals(self, V):
        """Unit normals out of the cells; the components towards the box
        faces fade out (not renormalised, so points near a face slow down
        instead of turning). Points on a face move within it."""
        n, _ = cell_normals(V, self.F, self.L, self.cell_ecs)
        n *= self.fade(V)
        on_box = self.fixed.any(1)
        ln = np.linalg.norm(n, axis=1)
        ok = on_box & (ln > 0.2)
        n[ok] /= ln[ok, None]
        n[on_box & ~ok] = 0
        n[~self.moving] = 0
        return n

    def displacement(self, d):
        """Displacement of the interface points by the offset flow."""
        off = self.offsets(d) / self.steps
        V = self.V0.copy()
        keep = ~self.moving | self.held
        for _ in range(self.steps):
            u = -off[:, None] * self.normals(V)
            u = taubin(u, self.S, self.smooth_iters, keep=keep)
            u[self.fixed] = 0
            V += u
            if self.pos_iters:
                V = taubin(V, self.S, self.pos_iters, keep=keep)
                V[self.fixed] = self.V0[self.fixed]
        return V - self.V0

    def points(self, d):
        return self.V0 + self.displacement(d)

    def share(self, d):
        return ecs_share(self.points(d), self.F, self.L)

    def solve(self, target, tol=1e-6, max_iter=40):
        """d with share(d) = target (Illinois variant of regula falsi)."""

        def f(d):
            return self.share(d) - target

        a, fa = 0.0, f(0.0)
        b = 2.0 if fa < 0 else -2.0
        fb = f(b)
        while fa * fb > 0:
            if abs(b) > 200:
                raise ValueError(f"ECS share {target} not reachable: {fb + target:.4f} at d={b}")
            a, fa = b, fb
            b *= 2
            fb = f(b)
        side = 0
        for _ in range(max_iter):
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            if abs(fc) < tol:
                break
            if fc * fb > 0:
                b, fb = c, fc
                if side == -1:
                    fa /= 2
                side = -1
            else:
                a, fa = c, fc
                if side == 1:
                    fb /= 2
                side = 1
        return c


def adjust_ecs_share(surf, target, max_rounds=30, verbose=True, **kwargs):
    """
    Move the cell-ECS interfaces of the multi-label surface surf so that the
    ECS takes the volume share target, keeping the box and a valid surface
    (no flipped or self-intersecting triangles). kwargs are passed to
    ECSAdjuster. Returns the new surface, the offset d and the achieved share.
    """
    adj = ECSAdjuster(surf, verbose=verbose, **kwargs)
    F, L, V0 = adj.F, adj.L, adj.V0
    strikes = np.zeros(len(V0), np.int64)
    for it in range(max_rounds):
        d = adj.solve(target)
        disp = adj.displacement(d)
        disp = extend_caps(V0, F, L, disp, adj.moving, adj.partner, pinned=strikes > 0)
        V = V0 + disp
        check = (np.linalg.norm(disp, axis=1) > 0)[F].any(1)
        bad = flipped_faces(V0, V, F) & check
        bad |= self_intersections(V, F, check)
        if verbose:
            print(f"round {it}: d={d:.3f}, share {ecs_share(V, F, L):.5f}, {bad.sum()} bad faces")
        if not bad.any():
            break
        # reduce the limits around the bad faces (to 0 after three strikes);
        # after two strikes they no longer take part in the smoothing, which
        # would spread the neighbours' motion into them
        bv = np.zeros(len(V0), bool)
        bv[F[bad].ravel()] = True
        bv = one_ring(bv, F)
        strikes[bv] += 1
        off = np.abs(adj.offsets(d))
        fac = np.where(strikes[bv] >= 3, 0.0, 0.5)
        adj.max_shrink[bv] = np.minimum(adj.max_shrink[bv], fac * off[bv])
        adj.max_grow[bv] = np.minimum(adj.max_grow[bv], fac * off[bv])
        adj.max_shrink, adj.max_grow = adj.limit(adj.max_shrink), adj.limit(adj.max_grow)
        adj.held |= strikes >= 2
    else:
        raise RuntimeError(f"no valid surface found in {max_rounds} rounds")
    out = surf.copy()
    out.points = V
    return out, d, ecs_share(V, F, L)
