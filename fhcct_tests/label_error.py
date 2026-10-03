"""Per-tet label error: fraction of the tet volume (random samples) whose winding-number label differs from the tet label."""

import sys

import numpy as np
import pyvista as pv

from emimesh.winding_number import label_points

f = sys.argv[1] if len(sys.argv) > 1 else "mid_labelled.vtu"
m = pv.read(f)
P, T, lab = m.points, m.cells_dict[10], m.cell_data["label"]
S = pv.read("../../emimesh/mid_0.15.vtk").clean()
rng = np.random.default_rng(0)
ns = 8
w = rng.dirichlet(np.ones(4), size=(len(T), ns))
X = np.einsum("tsk,tkd->tsd", w, P[T]).reshape(-1, 3)
L = label_points(S.points, S.regular_faces, S.cell_data["boundary_labels"], X).reshape(len(T), ns)
err = (L != lab[:, None]).mean(axis=1)
vol = m.compute_cell_sizes(length=False, area=False)["Volume"]
c = P[T].mean(axis=1)
dside = np.minimum.reduce([c[:, 0], 2000 - c[:, 0], c[:, 1], 2000 - c[:, 1]])
dcap = np.minimum(c[:, 2] - 950, 1050 - c[:, 2])
groups = {
    "near side (<30)": dside < 30,
    "near cap (<15)": (dcap < 15) & (dside >= 30),
    "interior": (dside >= 30) & (dcap >= 15),
}
print(f"{f}: mislabelled volume total {np.sum(err * vol) / vol.sum() * 100:.2f}%")
for g, msk in groups.items():
    print(
        f"  {g:16s}: tets {msk.sum():7d}, mislabelled volume {np.sum(err[msk] * vol[msk]) / vol[msk].sum() * 100:5.2f}%, "
        f"mean tet volume {vol[msk].mean():8.0f}"
    )
m.cell_data["label_error"] = err
m.save(f.replace(".vtu", "_err.vtu"))

# boundary faces: label just inside the face vs label of its tet
loc = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
tf = np.sort(T[:, loc].reshape(-1, 3), axis=1)
_, inv, cnt = np.unique(tf, axis=0, return_inverse=True, return_counts=True)
b = np.flatnonzero(cnt[inv.ravel()] == 1)
tet = b // 4
fc = P[tf[b]].mean(axis=1)
X = fc + 0.5 * (c[tet] - fc) / np.linalg.norm(c[tet] - fc, axis=1, keepdims=True)
Lb = label_points(S.points, S.regular_faces, S.cell_data["boundary_labels"], X)
fa = 0.5 * np.linalg.norm(np.cross(P[tf[b, 1]] - P[tf[b, 0]], P[tf[b, 2]] - P[tf[b, 0]]), axis=1)
oncap = (np.abs(fc[:, 2] - 950) < 3) | (np.abs(fc[:, 2] - 1050) < 3)
bad = Lb != lab[tet]
for g, msk in [("side walls", ~oncap), ("caps", oncap)]:
    print(
        f"  boundary {g:10s}: faces {msk.sum():6d}, mislabelled area {fa[msk & bad].sum() / fa[msk].sum() * 100:5.2f}%"
    )
