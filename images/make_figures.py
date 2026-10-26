"""
Figures of the readme, on a dense 2 um crop of the MICrONS segmentation:

    emi-download-data --cloudpath precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300 \\
        --position 162258-200928-20530 --size 2000 --mip 1 --output results/readme/raw.vti
    python images/make_figures.py results/readme/raw.vti results/readme [figures]

figures: any of pinches, handles, ecs, curved (default: all). The processed images,
surfaces and the ECS share adjustment are cached in the work directory.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from imagemesh.curved import compute_quadratic_quality, convert_to_quadratic
from imagemesh.facets import _tet_face_topology
from imagemesh.handles import fill_handles
from imagemesh.image import np2pv
from imagemesh.pinches import count_pinches, remove_pinches
from imagemesh.surface import label_volumes
from imagemesh.tetmesh import compute_surface_volume, mesh_surface
from matplotlib.collections import LineCollection

from emimesh.clients.generate_mesh import curve_mesh
from emimesh.clients.generate_surface_tags import mark_interfaces
from emimesh.ecs_share import adjust_ecs_share, ecs_share
from emimesh.extract_surfaces import extract_surface
from emimesh.process_image_data import process_image

pv.OFF_SCREEN = True
outdir = Path(__file__).parent
DX = 10
OPS = [
    ["removeislands", "minsize=5000"],
    ["dilate", "radius=1"],
    ["smooth", "iterations=1", "radius=1"],
    ["erode", "radius=1"],
]
ECS_CHANGE = -0.05  # change of the ECS share in the ECS figure
ZOOM = [1100, 1500, 1250, 1650]  # region of the zoomed ECS panel (nm)
CURVED_CROP = np.s_[0:50, 74:124, 30:50]  # 800 nm cube of the raw image
CURVED_DX = 20
CURVED_CELL = 19  # a dendritic spine (label after processing the crop)
CURVED_CLIM = (0, 8)  # nm
CURVED_CMAP = "YlOrRd"


def labels(grid):
    return np.asarray(grid["data"]).reshape(np.array(grid.dimensions) - 1, order="F")


def prepare(raw, work):
    """Processed image (OPS), after fill_handles, after remove_pinches, and
    their surfaces."""
    work.mkdir(parents=True, exist_ok=True)
    names = ["base", "filled", "repaired"]
    if not all((work / f"surf_{n}.vtk").exists() for n in names):
        grid, *_ = process_image(pv.read(raw), dx=DX, operations=OPS, num_threads=8)
        base = labels(grid)
        filled, fills = fill_handles(base, return_fills=True, workers=8)
        repaired = remove_pinches(filled)
        with open(work / "fills.pkl", "wb") as f:
            pickle.dump(fills, f)
        for n, img in zip(names, [base, filled, repaired]):
            print(f"{n}: {count_pinches(img)} pinched cubes", flush=True)
            g = np2pv(img, [DX] * 3)
            g.save(work / f"img_{n}.vti")
            extract_surface(g).save(work / f"surf_{n}.vtk")
    with open(work / "fills.pkl", "rb") as f:
        fills = pickle.load(f)
    return {n: pv.read(work / f"surf_{n}.vtk") for n in names}, fills


def cell_surface(surf, cell):
    L = surf.cell_data["boundary_labels"]
    return surf.extract_cells(np.flatnonzero((L == cell).any(1))).extract_surface(
        algorithm="dataset_surface"
    )


def slice_lines(surf, z):
    """2D segments of the slice of surf at height z."""
    sl = surf.slice("z", origin=(0, 0, z))
    if not sl.n_points:
        return np.zeros((0, 2, 2))
    return sl.points[sl.lines.reshape(-1, 3)[:, 1:]][:, :, :2]


# processing without erosion (as in the blockwise example), which leaves many
# pinches, e.g. along creases where a thin cell touches itself
PINCH_OPS = [
    ["removeislands", "minsize=5000"],
    ["dilate", "radius=1"],
    ["mode", "iterations=2"],
    ["smooth", "iterations=1", "radius=1"],
    ["upsample", "factor=2"],
    ["mode", "iterations=1"],
    ["zero_edges"],
    ["mode", "iterations=1"],
    ["zero_edges"],
]
PINCH_CELL = 24  # cell with a crease of pinches, shown around its densest part


def pinch_edges(img):
    """Midpoints and cells of the non-manifold edges of the cells' surfaces in
    the (unsmoothed) surface nets of the label image."""
    img = img.copy()
    img["data"][img["data"] == 0] = 1
    surf = img.contour_labels(
        "all", smoothing=False, output_mesh_type="triangles", background_value=0
    ).clean(tolerance=1e-6)
    F, L = surf.regular_faces, surf.cell_data["boundary_labels"]
    e = np.sort(np.stack([F, np.roll(F, -1, axis=1)], axis=2), axis=2).reshape(-1, 2)
    keys = []
    for side in range(2):
        lab = np.repeat(L[:, side], 3)
        keys.append(np.c_[lab, e][lab >= 2])
    keys, count = np.unique(np.vstack(keys), axis=0, return_counts=True)
    bad = keys[count > 2]
    return surf.points[bad[:, 1:]].mean(axis=1), bad[:, 0]


def pinches(raw, work, r=8 * DX, view=(-1.0, 1.0, 1.0)):
    """A cell around a crease of pinches (red points), before and after
    remove_pinches: its surface with a voxel plane across the crease (left),
    and that plane with the voxels, the cut of the surface (red) and the
    voxels changed by remove_pinches (outlined) (right)."""
    names = ["pinched", "unpinched"]
    if not all((work / f"surf_{n}.vtk").exists() for n in names):
        grid, *_ = process_image(pv.read(raw), dx=2 * DX, operations=PINCH_OPS, num_threads=8)
        imgs = [labels(grid), remove_pinches(labels(grid))]
        for n, img in zip(names, imgs):
            g = np2pv(img, [DX] * 3)
            g.save(work / f"img_{n}.vti")
            extract_surface(g).save(work / f"surf_{n}.vtk")
    imgs = {n: pv.read(work / f"img_{n}.vti") for n in names}
    mid, cells = pinch_edges(imgs["pinched"])
    print(f"pinch edges: {len(mid)} before, {len(pinch_edges(imgs['unpinched'])[0])} after")
    pts = mid[cells == PINCH_CELL]
    c = pts[np.argmax([(np.linalg.norm(pts - q, axis=1) < r).sum() for q in pts])]
    pts = pts[(np.abs(pts - c) < r).all(1)]
    box = [c[0] - r, c[0] + r, c[1] - r, c[1] + r, c[2] - r, c[2] + r]
    # voxel plane across the crease: the axis plane in the box with the most
    # 2x2 windows in which the cell touches itself only diagonally
    lo = ((c - r) // DX).astype(int)
    hi = ((c + r) // DX).astype(int)
    a, b = labels(imgs["pinched"]), labels(imgs["unpinched"])
    cell = a[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] == PINCH_CELL
    best = (-1, 0, 0)
    for axis in range(3):
        m = np.moveaxis(cell, axis, 0)
        p00, p11, p10, p01 = m[:, :-1, :-1], m[:, 1:, 1:], m[:, 1:, :-1], m[:, :-1, 1:]
        diag = (p00 & p11 & ~p10 & ~p01) | (p10 & p01 & ~p00 & ~p11)
        count = diag.sum(axis=(1, 2))
        if count.max() > best[0]:
            best = (count.max(), axis, lo[axis] + count.argmax())
    _, axis, layer = best
    other = [k for k in range(3) if k != axis]
    z = (layer + 0.5) * DX
    window = [slice(lo[k], hi[k]) for k in range(3)]
    window[axis] = layer
    ext = [lo[other[0]] * DX, hi[other[0]] * DX, lo[other[1]] * DX, hi[other[1]] * DX]
    plane = pv.Plane(
        center=np.where(np.arange(3) == axis, z, c),
        direction=np.eye(3)[axis],
        i_size=2 * r,
        j_size=2 * r,
    )
    view = np.asarray(view) / np.linalg.norm(view)
    changed = ((a != b) & ((a == PINCH_CELL) | (b == PINCH_CELL)))[tuple(window)].T

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for row, (name, title) in enumerate([("pinched", "before"), ("unpinched", "remove_pinches")]):
        cs = cell_surface(pv.read(work / f"surf_{name}.vtk"), PINCH_CELL)
        pl = pv.Plotter(window_size=(600, 600), off_screen=True)
        pl.add_mesh(
            cs.clip_box(box, invert=False), color="tab:orange", show_edges=True, line_width=0.3
        )
        pl.add_mesh(plane, color="tab:blue", opacity=0.35)
        if row == 0:
            pl.add_points(pts, color="red", point_size=8, render_points_as_spheres=True)
        pl.camera.focal_point = c
        pl.camera.position = c + 5 * r * view
        axs[row, 0].imshow(pl.screenshot(return_img=True))
        pl.close()
        axs[row, 0].set_title(f"{title}: surface", fontsize=10, loc="left")
        ax = axs[row, 1]
        sl = labels(imgs[name])[tuple(window)].T
        rgb = np.ones(sl.shape + (3,))
        rgb[sl >= 2] = 0.85
        rgb[sl == PINCH_CELL] = plt.matplotlib.colors.to_rgb("tab:orange")
        ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
        for t in np.arange(ext[0], ext[1] + DX, DX):
            ax.axvline(t, color="0.5", lw=0.3)
        for t in np.arange(ext[2], ext[3] + DX, DX):
            ax.axhline(t, color="0.5", lw=0.3)
        if row == 1:
            for j, i in np.argwhere(changed):
                x, y = ext[0] + i * DX, ext[2] + j * DX
                ax.add_patch(plt.Rectangle((x, y), DX, DX, fill=False, ec="k", lw=1.5))
        cut = cs.slice(normal=np.eye(3)[axis], origin=np.where(np.arange(3) == axis, z, c))
        if cut.n_points:
            seg = cut.points[cut.lines.reshape(-1, 3)[:, 1:]][:, :, other]
            ax.add_collection(LineCollection(seg, color="tab:red", lw=1.2))
        if row == 0:
            near = pts[np.abs(pts[:, axis] - z) <= DX]
            ax.plot(near[:, other[0]], near[:, other[1]], "o", color="red", ms=5)
        ax.set_xlim(ext[:2])
        ax.set_ylim(ext[2:])
        ax.set_title(f"{title}: voxel plane", fontsize=10, loc="left")
    for ax in axs.ravel():
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(outdir / "pinches.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# a cell with handles (well inside the crop), shown around its shortest closed
# handle
HANDLE_CELLS = [46]


def handles(surfs, fills):
    p = pv.Plotter(
        shape=(1, 2 * len(HANDLE_CELLS)), window_size=(800 * len(HANDLE_CELLS), 400), border=False
    )
    for col, cell in enumerate(HANDLE_CELLS):
        loop = min((f[1] for f in fills if f[0] == cell and f[2] is not None), key=len)
        loop = (np.asarray(loop) + 0.5) * DX
        c = loop.mean(0)
        normal = np.linalg.svd(loop - c)[2][2]
        r = 2.5 * np.linalg.norm(loop - c, axis=1).max()
        box = [c[0] - r, c[0] + r, c[1] - r, c[1] + r, c[2] - r, c[2] + r]
        for row, (name, title) in enumerate([("base", "before"), ("filled", "fill_handles")]):
            p.subplot(0, 2 * col + row)
            s = cell_surface(surfs[name], cell).clip_box(box, invert=False)
            p.add_mesh(s, color="lightsteelblue", smooth_shading=True)
            p.add_text(title, font_size=10)
            p.camera.focal_point = c
            p.camera.position = c + 4 * r * (normal + 0.5 * np.cross(normal, [0, 0, 1]))
    p.screenshot(outdir / "handles.png")
    p.close()


def ecs(surfs, work):
    surf = surfs["repaired"]
    V, F, L = surf.points, surf.regular_faces, surf.cell_data["boundary_labels"]
    share0 = ecs_share(V, F, L)
    path = work / f"surf_ecs_{ECS_CHANGE}.vtk"
    if not path.exists():
        adjust_ecs_share(surf, share0 + ECS_CHANGE)[0].save(path)
    adjusted = pv.read(path)
    share = ecs_share(adjusted.points, F, L)

    img = labels(pv.read(work / "img_repaired.vti"))
    k = img.shape[2] // 2
    z = (k + 0.5) * DX
    rng = np.random.default_rng(1)
    colors = plt.get_cmap("tab20")(rng.permutation(int(img.max()) + 1) % 20)
    rgb = colors[img[:, :, k].T]
    rgb[img[:, :, k].T == 0] = 1  # ECS white
    ext = [0, img.shape[0] * DX, 0, img.shape[1] * DX]
    lines = []
    for s, color, label in [
        (surf, "0.25", f"smoothed surface (ECS {share0:.0%})"),
        (adjusted, "tab:red", f"adjusted to {share:.0%} ECS"),
    ]:
        Ls = s.cell_data["boundary_labels"]
        iface = s.extract_cells(np.flatnonzero((Ls.min(1) == 1) & (Ls.max(1) >= 2)))
        lines.append(
            (slice_lines(iface.extract_surface(algorithm="dataset_surface"), z), color, label)
        )

    fig, axs = plt.subplots(1, 2, figsize=(9, 4.8))
    for ax, lim in zip(axs, [ext, ZOOM]):
        ax.imshow(rgb, origin="lower", extent=ext, alpha=0.45, interpolation="nearest")
        for seg, color, label in lines:
            ax.add_collection(LineCollection(seg, color=color, lw=0.8 if lim is ext else 1.5))
            ax.plot([], [], color=color, label=label)
        ax.set_xlim(lim[:2])
        ax.set_ylim(lim[2:])
        ax.set_aspect("equal")
        ax.set_axis_off()
    # voxel grid in the zoomed panel
    x0, x1, y0, y1 = ZOOM
    for t in np.arange(0, ext[1] + DX, DX):
        axs[1].axvline(t, color="0.6", lw=0.3, zorder=0.5)
    for t in np.arange(0, ext[3] + DX, DX):
        axs[1].axhline(t, color="0.6", lw=0.3, zorder=0.5)
    axs[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="k", lw=1)
    axs[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "ecs_share.png", dpi=120, bbox_inches="tight")


def quadratic_faces(P, F, level):
    """Surface of the 6-node triangles F, subdivided level times."""
    cells = np.column_stack([np.full(len(F), 6), F]).ravel()
    types = np.full(len(F), pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, types, P)
    return grid.extract_surface(algorithm="dataset_surface", nonlinear_subdivision=level)


def quadratic_edges(P, F, n=8):
    """The (curved) edges of the 6-node triangles F as polylines of n points."""
    E = np.vstack([F[:, [0, 3, 1]], F[:, [1, 4, 2]], F[:, [2, 5, 0]]])
    E = E[np.unique(np.sort(E[:, [0, 2]], axis=1), axis=0, return_index=True)[1]]
    t = np.linspace(0, 1, n)[:, None, None]
    a, m, b = P[E[:, 0]], P[E[:, 1]], P[E[:, 2]]
    X = (1 - t) * (1 - 2 * t) * a + 4 * t * (1 - t) * m + t * (2 * t - 1) * b
    lines = np.column_stack([np.full(len(E), n), np.arange(len(E) * n).reshape(n, -1).T])
    return pv.PolyData(X.reshape(-1, 3), lines=lines.ravel())


def curved(raw, work):
    """A dendritic spine of a small crop with straight (P1) and curved (P2) tets,
    colored by the distance to the input surface and crinkle-clipped inside the
    crop box, and the distance and quality of all membranes and tets of the crop."""
    work.mkdir(parents=True, exist_ok=True)
    paths = [work / f"curved_{n}" for n in ("surf.vtk", "p1.vtu", "p2.vtu")]
    if not all(p.exists() for p in paths):
        img = labels(pv.read(raw))[CURVED_CROP]
        ops = [["removeislands", "minsize=500"], ["smooth", "iterations=1", "radius=1"]]
        ops += [["erode", "radius=1"]]
        grid, *_ = process_image(np2pv(img, pv.read(raw).spacing), dx=CURVED_DX, operations=ops)
        surf = extract_surface(np2pv(remove_pinches(labels(grid)), [CURVED_DX] * 3))
        eps = 0.5 * CURVED_DX
        p1, _ = mesh_surface(surf, envelopsize=eps, simplify_eps=eps, edge_length_fac=0.06)
        for mesh, path in zip([surf, p1, curve_mesh(p1, surf)], paths):
            mesh.save(path)
    surf, p1, p2 = (pv.read(p) for p in paths)
    meshes = {"straight (P1)": convert_to_quadratic(p1), "curved (P2)": p2}

    cells = [c for c in np.unique(p1["label"]) if c >= 2]
    target = label_volumes(surf.points, surf.regular_faces, surf.cell_data["boundary_labels"])
    L = surf.cell_data["boundary_labels"]
    inner = surf.extract_cells(np.flatnonzero(L.min(1) >= 1))
    dist = {}
    for name, mesh in meshes.items():
        err = np.abs(compute_surface_volume(mesh, cells)[0] / target[cells] - 1)
        dist[name] = distance(membranes(mesh), inner)
        print(
            f"{name}: cell volume error mean {err.mean():.1%}, max {err.max():.1%}; "
            f"membrane distance mean {dist[name].mean():.2f} nm, "
            f"95th percentile {np.percentile(dist[name], 95):.1f} nm, max {dist[name].max():.1f} nm"
        )

    sheet = surf.extract_cells(np.flatnonzero((L == CURVED_CELL).any(1)))
    cell = p2.extract_cells(p2["label"] == CURVED_CELL)
    axes = np.linalg.svd(cell.points - cell.points.mean(axis=0), full_matrices=False)[2]
    proj = cell.points @ axes[0]
    if np.mean(proj) < 0.5 * (proj.min() + proj.max()):
        axes[0] *= -1  # head (the bulk of the points) on top
        proj = -proj
    length = np.ptp(proj)
    center = cell.points.mean(axis=0) + (0.5 * (proj.min() + proj.max()) - proj.mean()) * axes[0]
    head = center + 0.3 * length * axes[0]
    camera = dict(view=axes[2], up=axes[0])
    lo, hi = np.array(surf.bounds[::2]) + CURVED_DX, np.array(surf.bounds[1::2]) - CURVED_DX
    inside = lambda P: ((P > lo) & (P < hi)).all(axis=-1).all(axis=-1)  # noqa: E731
    input_cell = sheet.extract_surface(algorithm="dataset_surface")
    input_cell = input_cell.extract_cells(inside(input_cell.points[input_cell.regular_faces]))
    input_cell = input_cell.extract_surface(algorithm="dataset_surface")
    input_cell["distance (nm)"] = np.zeros(input_cell.n_points)
    full = [
        render(
            input_cell,
            input_cell.extract_all_edges(),
            center,
            0.55 * length,
            (600, 1400),
            1,
            **camera,
        )
    ]
    zoom = []
    for name, mesh in meshes.items():
        sub = mesh.extract_cells(mesh["label"] == CURVED_CELL)
        membrane = {tuple(sorted(f[:3])) for f in _tet_face_topology(sub)["boundary_faces"]}
        # crinkle clip: the whole tets inside the crop box, shrunk by one voxel
        tets = sub.cells_dict[pv.CellType.QUADRATIC_TETRA]
        kept = sub.extract_cells(inside(sub.points[tets[:, :4]]))
        F = _tet_face_topology(kept)["boundary_faces"]
        ids = np.asarray(kept["vtkOriginalPointIds"])
        cut = np.array([tuple(sorted(ids[f[:3]])) not in membrane for f in F])
        faces = quadratic_faces(kept.points, F[~cut], 4)
        faces["distance (nm)"] = distance(faces, sheet)
        extra = quadratic_faces(kept.points, F[cut], 4)
        edges = quadratic_edges(kept.points, F, 12)
        full.append(
            render(faces, edges, center, 0.55 * length, (600, 1400), 1.5, cut=extra, **camera)
        )
        zoom.append(
            render(
                faces, edges, head, 0.21 * length, (800, 800), 4, cut=extra, trim=False, **camera
            )
        )

    fig_w, fig_h, img_h = 11, 4.6, 0.8  # figure size (in) and image height (figure units)
    fig = plt.figure(figsize=(fig_w, fig_h))
    h = max(v.shape[0] for v in full)
    x = 0.0
    for v, title in zip(full, ["input surface", *meshes]):
        wi = v.shape[1] / h * img_h * fig_h / fig_w
        ax3d = fig.add_axes([x, 0.15, wi, img_h * v.shape[0] / h])
        ax3d.imshow(v)
        ax3d.set_axis_off()
        ax3d.set_title(title, fontsize=10)
        x += wi + 0.02
    zw = 0.5 * img_h * fig_h / fig_w  # square zooms, stacked
    for k, (v, title) in enumerate(zip(zoom, meshes)):
        axz = fig.add_axes([x, 0.15 + (1 - k) * 0.5 * img_h, zw, 0.5 * img_h - 0.01])
        axz.imshow(v)
        axz.set_xticks([])
        axz.set_yticks([])
        axz.set_ylabel(title.split()[0], fontsize=8)
        if k == 0:
            axz.set_title("spine head", fontsize=10)
    cax = fig.add_axes([0.03, 0.07, x + zw - 0.06, 0.025])
    sm = plt.cm.ScalarMappable(plt.Normalize(*CURVED_CLIM), CURVED_CMAP)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("distance to the input surface (nm)", fontsize=9)
    hist_x, hist_w = x + zw + 0.05, 0.97 - x - zw - 0.05
    ax = fig.add_axes([hist_x, 0.62, hist_w, 0.3])
    bins = np.linspace(0, CURVED_CLIM[1], 41)
    for (name, d), color in zip(dist.items(), ["tab:red", "tab:blue"]):
        label = f"{name}, mean {d.mean():.1f} nm"
        w = np.full(len(d), 1 / len(d))  # fractions of all samples, also those beyond the bins
        ax.hist(d, bins=bins, weights=w, histtype="stepfilled", alpha=0.4, color=color)
        ax.hist(d, bins=bins, weights=w, histtype="step", color=color, label=label)
    ax.set_xlabel("distance to the input surface (nm)", fontsize=9)
    ax.set_yticks([])
    ax.set_title("all membranes of the crop", fontsize=10)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_axes([hist_x, 0.15, hist_w, 0.3])
    bins = np.linspace(0, 1, 41)
    for (name, mesh), color in zip(meshes.items(), ["tab:red", "tab:blue"]):
        q = compute_quadratic_quality(mesh)
        label = f"{name}, mean {q.mean():.2f}, min {q.min():.2f}"
        print(f"{name}: quality {label.split(', ', 1)[1]}")
        w = np.full(len(q), 1 / len(q))
        ax.hist(q, bins=bins, weights=w, histtype="stepfilled", alpha=0.4, color=color)
        ax.hist(q, bins=bins, weights=w, histtype="step", color=color, label=label)
    ax.set_ylim(top=1.5 * ax.get_ylim()[1])
    ax.set_xlabel("cell quality (min. scaled Jacobian)", fontsize=9)
    ax.set_yticks([])
    ax.set_title("all tetrahedra of the crop", fontsize=10)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(outdir / "curved.png", dpi=150, bbox_inches="tight")


def render(faces, edges, focal, scale, size, line_width, view, up, cut=None, trim=True):
    """Parallel projection of faces colored by distance, with their edges and
    the (gray) faces of a clip."""
    p = pv.Plotter(window_size=size, border=False)
    p.add_mesh(
        faces,
        scalars="distance (nm)",
        cmap=CURVED_CMAP,
        clim=CURVED_CLIM,
        smooth_shading=True,
        show_scalar_bar=False,
    )
    if cut is not None and cut.n_cells:
        p.add_mesh(cut, color="lightgray", smooth_shading=True)
    p.add_mesh(edges, color="k", line_width=line_width)
    p.enable_parallel_projection()
    p.camera.focal_point = focal
    p.camera.position = focal + 10 * scale * view
    p.camera.up = up
    p.camera.parallel_scale = scale
    img = p.screenshot(return_img=True)
    p.close()
    if trim:
        rows, cols = np.nonzero((img < 250).any(axis=2))
        img = img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    return img


def distance(mesh, surf):
    """Distance of the points of mesh to surf."""
    _, closest = surf.find_closest_cell(mesh.points, return_closest_point=True)
    return np.linalg.norm(closest - mesh.points, axis=1)


def membranes(quad):
    """Interfaces between the regions of a quadratic mesh, subdivided."""
    f = mark_interfaces(quad, outer_offset=np.iinfo(np.int32).max // 2)
    f = f.extract_cells(f["boundaries"] < np.iinfo(np.int32).max // 2)
    return f.extract_surface(algorithm="dataset_surface", nonlinear_subdivision=4)


if __name__ == "__main__":
    raw, work = Path(sys.argv[1]), Path(sys.argv[2])
    figures = sys.argv[3:] or ["pinches", "handles", "ecs", "curved"]
    if {"handles", "ecs"} & set(figures):
        surfs, fills = prepare(raw, work)
    if "pinches" in figures:
        pinches(raw, work)
    if "handles" in figures:
        handles(surfs, fills)
    if "ecs" in figures:
        ecs(surfs, work)
    if "curved" in figures:
        curved(raw, work)
