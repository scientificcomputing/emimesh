import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

pv.OFF_SCREEN = True
bad = np.load("examples_pinches/bad_edges_without.npy")
good = np.load("examples_pinches/bad_edges_with.npy")
pinches = bad[cKDTree(good).query(bad)[0] > 1e-6]
print("pinch edges removed:", len(pinches))
cnt = np.array([len(x) for x in cKDTree(pinches).query_ball_point(pinches, 60)])
r = 150
surfs = {n: pv.read(f"examples_pinches/surf_{n}.vtk") for n in ("without", "with")}
s0 = surfs["without"]
for idx, ci in enumerate(np.argsort(cnt)[::-1][[0, 40, 200]]):
    c = pinches[ci]
    cc = s0.cell_centers().points
    label = s0.cell_data["boundary_labels"][np.argmin(np.linalg.norm(cc - c, axis=1))].max()
    p = pv.Plotter(shape=(1, 2), window_size=(1600, 800), border=False)
    for j, (name, s) in enumerate(surfs.items()):
        lab = s.cell_data["boundary_labels"]
        cell = s.extract_cells(np.flatnonzero((lab == label).any(axis=1))).extract_surface()
        cell = cell.clip_box([c[0] - r, c[0] + r, c[1] - r, c[1] + r, c[2] - r, c[2] + r], invert=False)
        p.subplot(0, j)
        p.add_mesh(cell, color="orange", show_edges=True, edge_color="k", line_width=0.5,
                   smooth_shading=False)
        p.add_points(pinches[np.linalg.norm(pinches - c, axis=1) < r], color="red",
                     point_size=8, render_points_as_spheres=True)
        p.add_text(("split_blocks ops" if name == "without" else "split_blocks ops + remove_pinches") + f" (cell {label})", font_size=12)
    p.link_views()
    p.camera.focal_point = c
    p.camera.position = c + np.array([1.0, 0.7, 0.5]) * 5 * r
    p.screenshot(f"examples_pinches/pinch_{idx}.png")
    p.close()
