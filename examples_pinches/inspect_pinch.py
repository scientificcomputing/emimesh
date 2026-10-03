import numpy as np, pyvista as pv
from scipy.spatial import cKDTree
from scipy import ndimage
bad = np.load("examples_pinches/bad_edges_without.npy"); good = np.load("examples_pinches/bad_edges_with.npy")
pinches = bad[cKDTree(good).query(bad)[0] > 1e-6]
cnt = np.array([len(x) for x in cKDTree(pinches).query_ball_point(pinches, 60)])
c = pinches[np.argsort(cnt)[::-1][40]]
ims = {n: pv.read(f"examples_pinches/img_{n}.vti") for n in ("without", "with")}
g = ims["without"]; dims = np.array(g.dimensions) - 1
A = {n: im["data"].reshape(dims, order="F") for n, im in ims.items()}
ijk = np.floor((c - np.array(g.origin)) / np.array(g.spacing)).astype(int)
print("center voxel", ijk, "spacing", g.spacing)
r = 15
sl = tuple(slice(max(i - r, 0), i + r) for i in ijk)
a, b = A["without"][sl], A["with"][sl]
diff = a != b
print("changed voxels in window:", diff.sum(), "total:", (A["without"] != A["with"]).sum())
print("changes old->new:", np.unique(np.stack([a[diff], b[diff]], 1), axis=0, return_counts=True))
# surface of cell 36 without: open boundary edges near c?
s = pv.read("examples_pinches/surf_without.vtk")
lab = s.cell_data["boundary_labels"]
cell = s.extract_cells(np.flatnonzero((lab == 36).any(axis=1))).extract_surface().clean()
fe = cell.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
near = np.linalg.norm(fe.points - c, axis=1) < 150
print("open boundary edge points of cell 36 surface near pinch:", near.sum(), "total", fe.n_points)
# print a z-slice of labels around center (1 = ECS, 36 = cell, other)
for name in ("without", "with"):
    m = A[name][ijk[0]-8:ijk[0]+8, ijk[1]-8:ijk[1]+8, ijk[2]]
    print(name); print("\n".join("".join("#" if v == 36 else "." if v == 1 or v == 0 else "o" for v in row) for row in m))
