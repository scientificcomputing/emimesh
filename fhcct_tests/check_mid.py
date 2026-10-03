"""
Check and label the dt mesh of build_mid.py. Band tets: winding number of the
compressed mid surface (closed, exact). Slab tets: the tubes close every cell region in
the slab, so each connected slab region (across non-constraint faces) is labelled from
the cap faces it touches (tetwild markers below/above, mid cap labels on the other side);
regions touching caps of several labels are reported.
"""

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from emimesh.winding_number import label_points

P0, F0 = np.load("mid_in_pts.npy"), np.load("mid_in_faces.npy")
meta = np.load("mid_in_meta.npz")
ntop, nbot, nwall, ntube = (int(meta[k]) for k in ["ntop", "nbot", "nwall", "ntube"])
bl, cap_lo, cap_hi, d = meta["bl"], meta["cap_lo"], meta["cap_hi"], float(meta["d"])
nS0 = ntop + nbot + nwall + ntube
m = pv.read("mid_out.vtk")
P, T = m.points.astype(np.float64), m.cells_dict[10]
key = lambda X: [tuple(r) for r in np.round(X, 6)]  # noqa: E731
idx = {k: i for i, k in enumerate(key(P))}
mp = np.array([idx.get(k, -1) for k in key(P0)])
print(f"points in/out {len(P0)} {len(P)}, tets {len(T)}, input points missing {(mp < 0).sum()}")
F = np.sort(mp[F0], axis=1)
loc = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
tf = np.sort(T[:, loc].reshape(-1, 3), axis=1)
fid = {}
for i, r in enumerate(map(tuple, tf)):
    fid.setdefault(r, []).append(i // 4)
present = np.array([tuple(r) in fid for r in F])
print(f"input tris present as tet faces: {present.sum()} / {len(F)}")
bnd = {r for r, t in fid.items() if len(t) == 1}
outer = {tuple(r) for r in F[: ntop + nbot + nwall]} | {
    tuple(r) for r in F[nS0:][(bl == 0).any(axis=1)]
}
print("mesh boundary faces:", len(bnd), " not caps/walls:", len(bnd - outer))
vol = m.compute_cell_sizes(length=False, area=False)["Volume"]
print("tets with volume<=0:", (vol <= 0).sum())

cc = P[T].mean(axis=1)
slab = (cc[:, 2] < 950.0 + d) | (cc[:, 2] > 1050.0 - d)
lab = np.zeros(len(T), dtype=int)
n_cap = int(meta["n_cap"])
lab[~slab] = label_points(P0[n_cap : n_cap + len(bl)], F0[nS0:] - n_cap, bl, cc[~slab])

# slab regions: connected across non-constraint faces
Fset = {tuple(r) for r in F}
_, inv = np.unique(tf, axis=0, return_inverse=True)
inv = inv.ravel()
o = np.argsort(inv, kind="stable")
pair = np.flatnonzero(inv[o][1:] == inv[o][:-1])
nb = np.column_stack([o[pair] // 4, o[pair + 1] // 4])
fk = tf[o[pair]]
is_c = np.array([tuple(r) in Fset for r in fk])
e = nb[~is_c & slab[nb].all(axis=1)]
_, comp = connected_components(
    coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(T), len(T))), directed=False
)
# seeds: tets on cap faces, weighted by face area
area = lambda f: (
    0.5 * np.linalg.norm(np.cross(P[f[:, 1]] - P[f[:, 0]], P[f[:, 2]] - P[f[:, 0]]), axis=1)
)  # noqa: E731
caps_lab = np.concatenate(
    [pv.read("top_cut.vtk").cell_data["marker"], pv.read("bottom_cut.vtk").cell_data["marker"]]
)
seed_t = [fid[tuple(f)][0] for f in F[: ntop + nbot]]
seed_l, seed_a = list(caps_lab), list(area(F[: ntop + nbot]))
FS = F[nS0:]
for f, L in zip(FS[cap_lo | cap_hi], bl[cap_lo | cap_hi].max(axis=1)):
    seed_t.append([t for t in fid[tuple(f)] if slab[t]][0])
    seed_l.append(L)
seed_a += list(area(FS[cap_lo | cap_hi]))
seed_t, seed_l, seed_a = np.array(seed_t), np.array(seed_l), np.array(seed_a)
sc = comp[seed_t]
key2, inv2 = np.unique(np.column_stack([sc, seed_l]), axis=0, return_inverse=True)
w = np.bincount(inv2.ravel(), seed_a)
best = {}
for (c, L), a in zip(key2, w):
    best.setdefault(c, []).append((a, L))
n_conf = 0
for c, lst in best.items():
    lst.sort(reverse=True)
    lab[(comp == c) & slab] = lst[0][1]
    if len(lst) > 1:
        n_conf += 1
        print(f"  slab region {c}: labels by cap area {[(int(L), round(a)) for a, L in lst]}")
print(f"slab regions with several cap labels: {n_conf}, unlabelled tets: {(lab == 0).sum()}")

jump = lab[nb[:, 0]] != lab[nb[:, 1]]
print(
    f"label jumps across non-constraint faces: {np.sum(jump & ~is_c)}, constraint faces without jump"
    f" (tubes+interfaces): {np.sum(~jump & is_c & ~np.isin(np.arange(len(nb)), []))}"
)
tube_set = {tuple(r) for r in F[ntop + nbot + nwall : nS0]}
is_tube = np.array([tuple(r) in tube_set for r in fk])
print(
    f"tube faces without label jump: {np.sum(is_tube & ~jump)} of {is_tube.sum()} (collars/tubes)"
)
for name, rng in [("top", slice(0, ntop)), ("bottom", slice(ntop, ntop + nbot))]:
    tc = np.array([fid[tuple(f)][0] for f in F[rng]])
    bad = lab[tc] != caps_lab[rng]
    a = area(F[rng])
    print(
        f"{name} cap: label != tetwild marker on {bad.sum()} faces, {a[bad].sum() / a.sum() * 100:.3f}%"
    )
m.cell_data["label"] = lab
m.cell_data["slab"] = slab.astype(np.uint8)
m.save("mid_labelled.vtu")
itf = pv.PolyData.from_regular_faces(P, fk[jump])
itf.cell_data["boundary_labels"] = np.sort(
    np.column_stack([lab[nb[jump, 0]], lab[nb[jump, 1]]]), axis=1
)
itf.cell_data["stitch"] = slab[nb[jump]].any(axis=1).astype(np.uint8)
itf.save("mid_interfaces.vtk")
