"""
Separation of touching cells by small local integer programs.

Every cell must be surrounded by extracellular space (ECS, label 0): no two
different cells may touch, at a face, an edge or a corner (26-neighbourhood).
separate_cells removes the contacts that a fast pass (nbmorph's
separate_labels) leaves, without the shortcuts that pass cannot avoid there:
no piece of a cell is cut off or deleted, every piece stays connected, and no
pinch is created. Topology is otherwise free to change (a tunnel may open or
close where that is the cheapest way out).

Each contact site is solved in a small box (radius R, plus a fixed margin of 2)
as a 0/1 program with HiGHS: the voxels within distance d of a contact are free,
each either keeps its piece (6-connected part of a label in the box) or becomes
ECS, and ECS voxels may join a nearby piece when that is needed (growth g).
Rows: separation (no two pieces of different cells 26-adjacent), pieces of one
cell do not merge, every piece stays connected (a unit flow from its largest
fixed part to each of its other fixed parts, plus cuts for detached free
voxels), and no new pinch (Latecki configurations, imagemesh.pinches, added
when a solution has one). Objective: the number of changed voxels. A heuristic
solution (contact voxels emptied, broken pieces reconnected by shortest paths
through ECS) is the starting point; a solution that reaches a simple lower
bound (one voxel of every touching pair) is final, otherwise it is polished once
with growth allowed. Boxes grow (R 3 -> 5) only when the small one has no
solution.

Blockwise: sites are taken per cell of a lattice of cell x cell voxels in
whole-image coordinates (origin), and the lattice cells are processed in the
8 colours of their index parities. Lattice cells of one colour are at least
cell voxels apart, more than a site reads (CROP) plus what a site changes
(WRITE), so they are independent and run in parallel (workers processes); the
sites of one lattice cell are solved one after another, in a fixed order.
HiGHS runs on one thread without a time limit, so every step is deterministic
and a block with a margin gets the same result as the whole image away from its
sides (time_limit is a safety stop only, and is reported when hit). Worker
processes are forked, so the caller needs no __main__ guard.
"""

import heapq
import itertools
import time
import warnings

import cc3d
import numba as nb
import numpy as np
from imagemesh.pinches import FACES, _cube_ok, _pinched_cubes, remove_pinches
from scipy import ndimage as ndi

O26 = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]
O6 = [o for o in O26 if sum(map(abs, o)) == 1]
CUBE = [(m & 1, (m >> 1) & 1, (m >> 2) & 1) for m in range(8)]  # k = x + 2y + 4z, as imagemesh
ST26 = ndi.generate_binary_structure(3, 3)
ST6 = ndi.generate_binary_structure(3, 1)
INF = float("inf")  # highspy.kHighsInf; highspy is imported where it is used, see _Model

# (R, d, grow, g) per stage: box radius, free distance from the contacts, growth of pieces into ECS
STAGES = [
    (3, 2, False, 0),
    (3, 2, True, 1),
    (3, 2, True, 2),
    (3, 3, True, 2),
    (5, 3, True, 2),
    (5, 5, True, 2),
]
MARGIN = 2  # fixed voxels around a box
HEUR_R = (3, 5, 8)  # box radii of the heuristic
CROP = max(HEUR_R) + MARGIN + 1  # a site reads voxels this close to its anchor
WRITE = max(s[0] for s in STAGES) + MARGIN  # and changes voxels this close


# --------------------------------------------------------------------------- helpers


@nb.njit(parallel=True, cache=True)
def _contact_mask(img, out):
    """out: voxels of a cell with a 26-neighbour of another cell."""
    nx, ny, nz = img.shape
    for i in nb.prange(nx):
        for j in range(ny):
            for k in range(nz):
                v = img[i, j, k]
                hit = False
                if v != 0:
                    for a in range(max(i - 1, 0), min(i + 2, nx)):
                        for b in range(max(j - 1, 0), min(j + 2, ny)):
                            for c in range(max(k - 1, 0), min(k + 2, nz)):
                                w = img[a, b, c]
                                if w != 0 and w != v:
                                    hit = True
                out[i, j, k] = hit


def contact_mask(img):
    """Voxels of a cell that touch another cell (26-neighbourhood)."""
    out = np.empty(img.shape, np.bool_)
    _contact_mask(np.ascontiguousarray(img), out)
    return out


def _bounds(center, R, M, sh):
    lo = [max(c - R, 0) for c in center]
    hi = [min(c + R + 1, s) for c, s in zip(center, sh)]
    clo = [max(a - M, 0) for a in lo]
    chi = [min(b + M, s) for b, s in zip(hi, sh)]
    return lo, hi, clo, chi


def _box(img, center, R, M):
    """The box of radius R plus margin M around center: slice, and mask of the inner box."""
    lo, hi, clo, chi = _bounds(center, R, M, img.shape)
    sl = tuple(slice(a, b) for a, b in zip(clo, chi))
    inner = np.zeros([b - a for a, b in zip(clo, chi)], bool)
    inner[tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, clo))] = True
    return sl, inner


def _shifted(c):
    """The 26 shifted views of c (padded with 0)."""
    P = np.pad(c, 1)
    s = P.shape
    for dz, dy, dx in O26:
        yield P[1 + dz : s[0] - 1 + dz, 1 + dy : s[1] - 1 + dy, 1 + dx : s[2] - 1 + dx]


def _touch(c):
    t = np.zeros(c.shape, bool)
    for w in _shifted(c):
        t |= (c > 0) & (w > 0) & (w != c)
    return t


def _foreign_near(c, label):
    t = np.zeros(c.shape, bool)
    for w in _shifted(c):
        t |= (w > 0) & (w != label)
    return t


def _n6(c, label):
    return cc3d.connected_components((c == label).astype(np.uint8), connectivity=6, return_N=True)[
        1
    ]


def _inb(q, S):
    return all(0 <= q[i] < S[i] for i in range(3))


# --------------------------------------------------------------------------- heuristic


def _heuristic_box(img, center, R, order=1, core=0):
    """Empty the contact voxels of the box (and those within core of them), reconnect
    every piece split by that through ECS voxels that touch no other cell (shortest
    paths), restore a voxel of a piece that vanished, and repair new pinches. Applied
    to img in place if the box then has no contact, the same number of pieces per cell
    and no more pinched cubes; returns whether it was."""
    sl, inner = _box(img, center, R, MARGIN)
    ctx = img[sl]
    orig = ctx.copy()
    labs = [int(v) for v in np.unique(orig[inner]) if v]
    n0 = {lab: _n6(orig, lab) for lab in labs}
    cc0 = {
        lab: cc3d.connected_components((orig == lab).astype(np.uint8), connectivity=6)
        for lab in labs
    }
    t0 = _touch(orig) & inner
    c = orig.copy()
    c[t0] = 0
    if core:
        c[ndi.binary_dilation(t0, ST26, iterations=core) & inner] = 0
    S = c.shape
    for lab in sorted(labs, key=lambda v: (order * np.count_nonzero(orig == v), v)):
        while True:
            m = c == lab
            cc1, n1 = cc3d.connected_components(m.astype(np.uint8), connectivity=6, return_N=True)
            lost = [o for o in range(1, int(cc0[lab].max()) + 1) if not np.any(m & (cc0[lab] == o))]
            if lost:  # a whole piece vanished: restore one voxel of it
                q = np.argwhere((cc0[lab] == lost[0]) & inner)
                if not len(q):
                    return False
                c[tuple(q[0])] = lab
                continue
            need = None  # two new parts that belonged to one piece
            for k in range(1, n1 + 1):
                ok = np.unique(cc0[lab][cc1 == k])
                ok = ok[ok > 0]
                for k2 in range(k + 1, n1 + 1):
                    if np.isin(np.unique(cc0[lab][cc1 == k2]), ok).any():
                        need = (k, k2)
                        break
                if need:
                    break
            if need is None:
                break
            free = (c == 0) & inner & ~_foreign_near(c, lab)
            src, dst = cc1 == need[0], cc1 == need[1]
            dist, prev, pq = {}, {}, []
            for q in map(tuple, np.argwhere(free)):
                if any(
                    _inb(w := (q[0] + o[0], q[1] + o[1], q[2] + o[2]), S) and src[w] for o in O6
                ):
                    dist[q] = 1
                    heapq.heappush(pq, (1, q))
            end = None
            while pq:
                d, q = heapq.heappop(pq)
                if d > dist.get(q, 1e9):
                    continue
                if any(
                    _inb(w := (q[0] + o[0], q[1] + o[1], q[2] + o[2]), S) and dst[w] for o in O6
                ):
                    end = q
                    break
                for o in O6:
                    w = (q[0] + o[0], q[1] + o[1], q[2] + o[2])
                    if _inb(w, S) and free[w] and d + 1 < dist.get(w, 1e9):
                        dist[w] = d + 1
                        prev[w] = q
                        heapq.heappush(pq, (d + 1, w))
            if end is None:
                return False
            q = end
            while True:
                c[q] = lab
                if q not in prev:
                    break
                q = prev[q]
    np0 = _pinched_cubes(np.ascontiguousarray(orig)).sum()
    if _pinched_cubes(np.ascontiguousarray(c)).sum() > np0:
        r = remove_pinches(np.ascontiguousarray(c), verbose=False)
        r[~inner] = orig[~inner]
        c = r
    if (_touch(c) & inner).any() or any(_n6(c, lab) != n0[lab] for lab in labs):
        return False
    if _pinched_cubes(np.ascontiguousarray(c)).sum() > np0:
        return False
    ctx[...] = c
    return True


def _heuristic(img, center):
    """Heuristic solution of the site (on a copy of img), or None."""
    W = img.copy()
    for R in HEUR_R:
        for core in (0, 1, 2, 3):
            for order in (1, -1):
                if _heuristic_box(W, center, R, order, core):
                    return W
    return None


# --------------------------------------------------------------------------- integer program


class _Model:
    def __init__(self, time_limit):
        # imported here: the PyPI wheel of highspy, imported before numba has compiled
        # anything, can make llvmlite fail to load (conda-forge's build does not)
        import highspy

        self.hs = highspy
        self.h = highspy.Highs()
        self.h.silent()
        self.h.setOptionValue("time_limit", float(time_limit))
        self.h.setOptionValue("threads", 1)
        self.h.setOptionValue("mip_abs_gap", 0.99)  # the objective is integer
        # presolve costs more than it saves on these small models (2-3x on the slow sites)
        self.h.setOptionValue("presolve", "off")
        self.n = 0

    def cols(self, k, lo, up, cost=None, integer=False):
        start = self.n
        self.h.addVars(k, np.full(k, lo, float), np.full(k, up, float))
        idx = np.arange(start, start + k, dtype=np.int32)
        if integer:
            kint = self.hs.HighsVarType.kInteger
            self.h.changeColsIntegrality(k, idx, np.array([kint] * k))
        if cost is not None:
            self.h.changeColsCost(k, idx, np.asarray(cost, float))
        self.n += k
        return start

    def row(self, idx, val, lo, up):
        self.h.addRow(lo, up, len(idx), np.asarray(idx, np.int32), np.asarray(val, float))


def _pinch_rows(cu, ctx, free, dom, var, plab):
    """Rows forbidding the pinch configurations (imagemesh _cube_ok) of one cube."""
    vox = [(cu[0] + a, cu[1] + b, cu[2] + c) for a, b, c in CUBE]
    vals = {0}
    for q in vox:
        vals |= {int(plab[k]) for k in dom[q]} if free[q] else {int(ctx[q])}

    def lit(q, lab):  # [q has value lab] = const + sum coef * x
        if not free[q]:
            return (1.0 if ctx[q] == lab else 0.0), {}
        if lab == 0:
            return 1.0, {var[q, k]: -1.0 for k in dom[q]}
        return 0.0, {var[q, k]: 1.0 for k in dom[q] if plab[k] == lab}

    out = []

    def row(pos, neg, rhs):  # sum pos - sum neg <= rhs
        c0, coef = 0.0, {}
        for terms, sgn in ((pos, 1.0), (neg, -1.0)):
            for q, lab in terms:
                c, dd = lit(q, lab)
                c0 += sgn * c
                for j, a in dd.items():
                    coef[j] = coef.get(j, 0.0) + sgn * a
        coef = {j: a for j, a in sorted(coef.items()) if a}
        if c0 + sum(a for a in coef.values() if a > 0) <= rhs + 1e-9:
            return  # never violated
        out.append((list(coef), list(coef.values()), -INF, rhs - c0))

    for lab in sorted(vals):
        for f in FACES:
            a, b, c, d = (vox[i] for i in f)
            row([(a, lab), (c, lab)], [(b, lab), (d, lab)], 1.0)
            row([(b, lab), (d, lab)], [(a, lab), (c, lab)], 1.0)
        for k in range(4):
            c1, c2 = vox[k], vox[7 - k]
            rest = [vox[i] for i in range(8) if i not in (k, 7 - k)]
            row([(c1, lab), (c2, lab)], [(r, lab) for r in rest], 1.0)
            row([(r, lab) for r in rest], [(c1, lab), (c2, lab)], 5.0)
    return out


def _flow_rows(md, k, fq, var, fcc, m, free, S):
    """Connectivity of piece k: free voxels fq, fixed parts 1..m of fcc (one node each)."""
    node = {q: i for i, q in enumerate(fq)}
    nodes = len(fq) + m
    edges = set()
    for q in fq:
        for o in O6:
            w = (q[0] + o[0], q[1] + o[1], q[2] + o[2])
            if not _inb(w, S):
                continue
            if w in node:
                if node[w] > node[q]:
                    edges.add((node[q], node[w]))
            elif fcc[w]:
                edges.add((node[q], len(fq) + int(fcc[w]) - 1))
    E = sorted(edges)

    def arcs(cap):
        f0 = md.cols(2 * len(E), 0.0, cap)
        inc = [[] for _ in range(nodes)]
        for e, (a, b) in enumerate(E):
            fab, fba = f0 + 2 * e, f0 + 2 * e + 1
            inc[b] += [(fab, 1.0), (fba, -1.0)]
            inc[a] += [(fab, -1.0), (fba, 1.0)]
        return f0, inc

    if m:  # a unit commodity from the largest fixed part to every other one, through kept voxels
        root = len(fq) + int(np.argmax(np.bincount(fcc.ravel())[1:]))
        for t in range(len(fq), nodes):
            if t == root:
                continue
            f0, inc = arcs(1.0)
            for e, (a, b) in enumerate(E):
                for v in (a, b):
                    if v < len(fq):
                        md.row(
                            [f0 + 2 * e, f0 + 2 * e + 1, var[fq[v], k]], [1.0, 1.0, -1.0], -INF, 0.0
                        )
            for v in range(nodes):
                rhs = -1.0 if v == root else (1.0 if v == t else 0.0)
                md.row([c for c, _ in inc[v]], [s for _, s in inc[v]], rhs, rhs)
        return None
    # the piece is all free: a root voxel sends one unit to every kept voxel
    f0, inc = arcs(float(nodes))
    for e, (a, b) in enumerate(E):
        for c in (f0 + 2 * e, f0 + 2 * e + 1):
            for v in (a, b):
                md.row([c, var[fq[v], k]], [1.0, -float(nodes)], -INF, 0.0)
    xs = [var[q, k] for q in fq]
    r0 = md.cols(len(fq), 0.0, 1.0, integer=True)
    s0 = md.cols(len(fq), 0.0, float(nodes))
    md.row(list(range(r0, r0 + len(fq))), [1.0] * len(fq), 1.0, 1.0)
    md.row(xs, [1.0] * len(xs), 1.0, INF)
    md.row(list(range(s0, s0 + len(fq))) + xs, [1.0] * len(fq) + [-1.0] * len(xs), 0.0, 0.0)
    for v, q in enumerate(fq):
        md.row([r0 + v, var[q, k]], [1.0, -1.0], -INF, 0.0)
        md.row([s0 + v, r0 + v], [1.0, -float(nodes)], -INF, 0.0)
        md.row(
            [c for c, _ in inc[v]] + [s0 + v, var[q, k]],
            [s for _, s in inc[v]] + [1.0, -1.0],
            0.0,
            0.0,
        )
    return (k, r0, fq)


def _mip_box(img, center, R, d, grow, g, warm=None, time_limit=60.0, max_rounds=50):
    """Solve one box of img in place (see the module docstring). Returns a dict with
    status ('optimal', 'limit': time limit hit with a solution, 'nothing': no contact,
    or HiGHS' status) and the number of changed voxels."""
    import highspy

    sl, inner = _box(img, center, R, MARGIN)
    ctx = img[sl].copy()
    S = ctx.shape
    info = dict(R=R, d=d, g=g if grow else 0)
    tm = _touch(ctx) & inner
    if not tm.any():
        return dict(status="nothing", changed=0, **info)
    free = (ndi.binary_dilation(tm, ST26, iterations=d) & inner) if d else tm.copy()
    pc = cc3d.connected_components(ctx, connectivity=6)  # pieces
    plab = np.zeros(int(pc.max()) + 1, np.int64)
    plab[pc.ravel()] = ctx.ravel()
    wpiece = None
    if warm is not None:  # the warm start's voxels, as pieces of ctx
        wctx = warm[sl]
        diff = wctx != ctx
        if diff.any() and not (diff & ~inner).any():
            free |= diff
            wcc = cc3d.connected_components(wctx, connectivity=6)
            same = (wctx == ctx) & (pc > 0)
            votes = {}
            for c_, p_ in zip(wcc[same].ravel().tolist(), pc[same].ravel().tolist()):
                votes.setdefault(c_, {}).setdefault(p_, 0)
                votes[c_][p_] += 1
            wpiece = np.zeros(S, np.int64)
            for c_, cnt in votes.items():
                wpiece[(wcc == c_) & free] = max(sorted(cnt), key=cnt.get)
            wpiece[(wctx == 0) | ~free] = 0
            if ((wctx > 0) & free & (wpiece == 0)).any():
                wpiece = None  # a part not attached to any piece: no warm start
    cand = {}
    if grow:
        reach = ndi.binary_dilation(free, ST26, iterations=g + 1)
        for k in np.unique(pc[reach & (pc > 0)]).tolist():
            cand[k] = ndi.binary_dilation(pc == k, ST26, iterations=g) & free & (ctx == 0)
    fl = [tuple(int(v) for v in q) for q in np.argwhere(free)]
    dom, var, keys = {}, {}, []
    for q in fl:
        ks = [int(pc[q])] if ctx[q] else [k for k, mk in cand.items() if mk[q]]
        if wpiece is not None and wpiece[q] and int(wpiece[q]) not in ks:
            ks.append(int(wpiece[q]))
        dom[q] = ks
        for k in ks:
            var[q, k] = len(keys)
            keys.append((q, k))
    n = len(keys)
    md = _Model(time_limit)
    md.cols(
        n, 0.0, 1.0, [-1.0 if ctx[q] else 1.0 for q, k in keys], integer=True
    )  # changed = const + cost x
    zero = set()
    for q in fl:
        if len(dom[q]) > 1:
            md.row([var[q, k] for k in dom[q]], [1.0] * len(dom[q]), -INF, 1.0)
        for k in dom[q]:
            la = plab[k]
            for o in O26:
                w = (q[0] + o[0], q[1] + o[1], q[2] + o[2])
                if not _inb(w, S):
                    continue
                face = sum(map(abs, o)) == 1
                if free[w]:
                    if w < q:
                        continue
                    for k2 in dom[w]:
                        if plab[k2] != la or (face and k2 != k):
                            md.row([var[q, k], var[w, k2]], [1.0, 1.0], -INF, 1.0)
                else:
                    k2 = int(pc[w])
                    if k2 and (plab[k2] != la or (face and k2 != k)):
                        zero.add(var[q, k])
    for j in sorted(zero):
        md.h.changeColBounds(j, 0.0, 0.0)
    # a kept free voxel of a piece with fixed voxels needs a face neighbour of the piece
    for q in fl:
        for k in dom[q]:
            nb_, fixed_nb = [], False
            for o in O6:
                w = (q[0] + o[0], q[1] + o[1], q[2] + o[2])
                if not _inb(w, S):
                    continue
                if free[w]:
                    if (w, k) in var:
                        nb_.append(var[w, k])
                elif pc[w] == k:
                    fixed_nb = True
            if not fixed_nb and ((pc == k) & ~free).any():
                md.row(nb_ + [var[q, k]], [1.0] * len(nb_) + [-1.0], 0.0, INF)
    roots, flow_pieces = [], []
    for k in sorted({k for _, k in keys}):
        fcc, m = cc3d.connected_components(
            ((pc == k) & ~free).astype(np.uint8), connectivity=6, return_N=True
        )
        fq = [q for q in fl if (q, k) in var]
        if m + len(fq) <= 1:
            if m == 0 and fq:  # a one-voxel piece, all free: it must stay
                md.row([var[fq[0], k]], [1.0], 1.0, INF)
            continue
        r = _flow_rows(md, k, fq, var, fcc, m, free, S)
        if r is None:
            flow_pieces.append(k)
        else:
            roots.append(r)
    sol = None
    if wpiece is not None:
        x0 = np.zeros(md.n)
        for (q, k), j in var.items():
            x0[j] = 1.0 if wpiece[q] == k else 0.0
        for k, r0, fq in roots:
            for v, q in enumerate(fq):
                if wpiece[q] == k:
                    x0[r0 + v] = 1.0
                    break
        sol = highspy.HighsSolution()
        sol.col_value = list(x0)
        sol.value_valid = True
    pin0 = _pinched_cubes(np.ascontiguousarray(ctx))
    cubes = sorted({(q[0] - a, q[1] - b, q[2] - c) for q in fl for a, b, c in CUBE})
    cubes = [cu for cu in cubes if all(0 <= cu[i] < S[i] - 1 for i in range(3)) and not pin0[cu]]
    added = set()
    limited = False
    for rnd in range(max_rounds):
        if sol is not None:
            md.h.setSolution(sol)
        md.h.run()
        ms = md.h.getModelStatus()
        if ms == highspy.HighsModelStatus.kTimeLimit and md.h.getInfo().primal_solution_status == 2:
            ms, limited = highspy.HighsModelStatus.kOptimal, True
        if ms != highspy.HighsModelStatus.kOptimal:
            return dict(status=md.h.modelStatusToString(ms), **info)
        x = np.round(np.array(md.h.getSolution().col_value[:n])).astype(int)
        new = ctx.copy()
        newp = np.where(free, 0, pc)
        for q in fl:
            new[q] = 0
        for j in np.flatnonzero(x):
            q, k = keys[j]
            new[q] = plab[k]
            newp[q] = k
        nr = 0
        for k in flow_pieces:  # free voxels kept but detached from the piece's fixed parts
            mk = newp == k
            cc, nc = cc3d.connected_components(mk.astype(np.uint8), connectivity=6, return_N=True)
            if nc <= 1:
                continue
            fixed_cc = set(np.unique(cc[mk & ~free]).tolist())
            for c_ in range(1, nc + 1):
                if c_ in fixed_cc:
                    continue
                A = cc == c_
                bd = ndi.binary_dilation(A, ST6) & ~A
                bcols = [
                    var[t, k]
                    for t in map(lambda v: tuple(int(i) for i in v), np.argwhere(bd))
                    if (t, k) in var
                ]
                for q in np.argwhere(A):
                    md.row(
                        bcols + [var[tuple(int(i) for i in q), k]],
                        [1.0] * len(bcols) + [-1.0],
                        0.0,
                        INF,
                    )
                    nr += 1
        for cu in cubes:
            if cu in added:
                continue
            v = np.array([new[cu[0] + a, cu[1] + b, cu[2] + c] for a, b, c in CUBE], dtype=np.int64)
            if (v == v[0]).all() or _cube_ok(v):
                continue
            added.add(cu)
            for r in _pinch_rows(cu, ctx, free, dom, var, plab):
                md.row(*r)
                nr += 1
        if not nr:
            img[sl] = new
            return dict(
                status="limit" if limited else "optimal",
                changed=int(np.count_nonzero(new != ctx)),
                **info,
            )
    return dict(status="round limit", **info)


def contact_lower_bound(img, center, R=3):
    """Lower bound on the changed voxels of a box (with any growth): of every pair of
    touching voxels of different cells one becomes ECS. Voxels touching a fixed voxel of
    another cell are forced; a maximum matching of the other pairs adds one per pair."""
    import networkx as nx

    sl, inner = _box(img, center, R, MARGIN)
    ctx = img[sl]
    S = ctx.shape
    edges, forced = [], set()
    for q in map(tuple, np.argwhere(inner & _touch(ctx))):
        for o in O26:
            w = (q[0] + o[0], q[1] + o[1], q[2] + o[2])
            if not _inb(w, S) or ctx[w] == 0 or ctx[w] == ctx[q]:
                continue
            if inner[w]:
                if q < w:
                    edges.append((q, w))
            else:
                forced.add(q)
    G = nx.Graph([e for e in edges if e[0] not in forced and e[1] not in forced])
    return len(forced) + len(nx.max_weight_matching(G, maxcardinality=True))


def _solve_site(img, center, time_limit=60.0, polish=True):
    """Solve the site at center in img (in place): the stages in order until one has a
    solution, then a polish with growth if the solution is above the lower bound."""
    W = _heuristic(img, center)
    for R, d, grow, g in STAGES:
        before = img.copy() if polish and not grow else None
        r = _mip_box(img, center, R, d, grow, g, warm=W, time_limit=time_limit)
        if r["status"] not in ("optimal", "limit", "nothing"):
            continue
        if (
            before is not None
            and r["status"] != "nothing"
            and r["changed"] > contact_lower_bound(before, center, R)
        ):
            P = before
            r2 = _mip_box(P, center, R, d, True, 1, warm=img, time_limit=time_limit)
            if r2["status"] in ("optimal", "limit") and r2["changed"] < r["changed"]:
                img[...] = P
                r = r2
        return r
    return dict(status="failed")


# --------------------------------------------------------------------------- driver


def _lattice_task(args):
    """Solve the sites of one lattice cell. win: the image around it, (clo, chi): the
    lattice cell in win, goff: whole-image position of win[0, 0, 0]."""
    win, clo, chi, goff, time_limit, polish = args
    t0 = time.perf_counter()
    old = win.copy()
    sl = tuple(slice(a, b) for a, b in zip(clo, chi))
    mk = contact_mask(win)[sl]
    cc, n = cc3d.connected_components(mk.astype(np.uint8), connectivity=26, return_N=True)
    anchors = []
    for i in range(1, n + 1):
        g = np.argwhere(cc == i).mean(0) + np.asarray(clo) + np.asarray(goff)
        anchors.append(tuple(int(v) for v in np.floor(g + 0.5) - np.asarray(goff)))
    out = []
    for ce in sorted(anchors):
        lo, hi, _, _ = _bounds(ce, CROP, 0, win.shape)
        csl = tuple(slice(a, b) for a, b in zip(lo, hi))
        sub = win[csl].copy()
        r = _solve_site(sub, tuple(c - a for c, a in zip(ce, lo)), time_limit, polish)
        win[csl] = sub
        out.append(r)
    ch = np.argwhere(win != old)
    return ch, win[tuple(ch.T)], out, time.perf_counter() - t0


def _init_worker():
    nb.set_num_threads(1)


def separate_cells(
    img,
    cell=20,
    max_rounds=4,
    workers=1,
    polish=True,
    time_limit=60.0,
    origin=(0, 0, 0),
    verbose=True,
    return_stats=False,
):
    """
    Separate all touching cells of img (0 = ECS) with every piece of every cell kept
    and connected and no new pinch, changing as few voxels as possible per site; see
    the module docstring. Meant after nbmorph's separate_labels, for the contacts it
    leaves. Returns a separated copy (and a list of per-round stats, if return_stats).

    Args:
        img (np.ndarray): 3D label image.
        cell (int): Lattice cell size in voxels (>= CROP + WRITE = 18). Sites are
            taken per lattice cell; same-colour cells run in parallel.
        max_rounds (int): Passes over the 8 colours (stops when no contact is left).
        workers (int): Processes for the lattice cells of one colour.
        polish (bool): Re-solve with growth when a solution is above the lower bound.
        time_limit (float): Safety stop per HiGHS run in seconds; a run that hits it
            is reported, since its result can then depend on the machine.
        origin (tuple): Position of img[0, 0, 0] in the whole image (blockwise).
        verbose (bool): Print a line per round.
        return_stats (bool): Also return the per-round stats.
    """
    if cell < CROP + WRITE:
        raise ValueError(f"cell must be at least {CROP + WRITE}")
    img = np.array(img, copy=True, order="C")
    o = np.asarray(origin, np.int64)
    pool = None
    if workers > 1:
        import multiprocessing as mp

        # fork: workers inherit the loaded modules and compiled functions and never
        # re-run the caller's script (spawn would, without a __main__ guard)
        pool = mp.get_context("fork").Pool(workers, initializer=_init_worker)
    stats = []
    try:
        for rnd in range(max_rounds):
            t0 = time.perf_counter()
            mk = contact_mask(img)
            if not mk.any():
                break
            cid = np.unique((np.argwhere(mk) + o) // cell, axis=0)
            st = dict(round=rnd, contacts=int(mk.sum()), cells=len(cid), sites=0, changed=0)
            for colour in range(8):
                par = np.array([(colour >> 2) & 1, (colour >> 1) & 1, colour & 1])
                sel = cid[(cid % 2 == par).all(1)]
                tasks = []
                for c in sel:
                    clo = np.maximum(c * cell - o, 0)
                    chi = np.minimum((c + 1) * cell - o, img.shape)
                    wlo = np.maximum(clo - CROP, 0)
                    whi = np.minimum(chi + CROP, img.shape)
                    wsl = tuple(slice(a, b) for a, b in zip(wlo, whi))
                    tasks.append(
                        (
                            wsl,
                            (
                                img[wsl].copy(),
                                tuple(clo - wlo),
                                tuple(chi - wlo),
                                tuple(wlo + o),
                                time_limit,
                                polish,
                            ),
                        )
                    )
                res = (
                    pool.imap(_lattice_task, [t for _, t in tasks])
                    if pool
                    else map(_lattice_task, [t for _, t in tasks])
                )
                for (wsl, _), (ch, vals, out, _t) in zip(tasks, res):
                    win = img[wsl]
                    win[tuple(ch.T)] = vals
                    st["changed"] += len(ch)
                    for r in out:
                        st["sites"] += 1
                        k = (
                            r["status"]
                            if r["status"] in ("nothing", "failed")
                            else f"R{r['R']} g{r['g']} {r['status']}"
                        )
                        st[k] = st.get(k, 0) + 1
            st["touching"] = int(contact_mask(img).sum())
            st["time"] = round(time.perf_counter() - t0, 2)
            stats.append(st)
            if verbose:
                print(
                    f"separate_cells, round {rnd}: "
                    + ", ".join(f"{k} {v}" for k, v in st.items() if k != "round")
                )
            if any("limit" in k and "round" not in k for k in st):
                warnings.warn(
                    "separate_cells: a HiGHS run hit time_limit; "
                    "blocks may then differ from the whole image"
                )
    finally:
        if pool:
            pool.close()
            pool.join()
    return (img, stats) if return_stats else img
