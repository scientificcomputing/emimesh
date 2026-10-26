"""
Figures of the readme, on a dense 2 um crop of the MICrONS segmentation:

    emi-download-data --cloudpath precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300 \\
        --position 162258-200928-20530 --size 2000 --mip 1 --output results/readme/raw.vti
    python images/make_figures.py results/readme/raw.vti results/readme [figures]

figures: any of pinches, handles, ecs (default: all). The processed images,
surfaces and the ECS share adjustment are cached in the work directory.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from imagemesh.handles import fill_handles
from imagemesh.image import np2pv
from imagemesh.pinches import count_pinches, remove_pinches
from matplotlib.collections import LineCollection

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
# handle; handles of cells cut by the crop faces can be artefacts of the cut
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


if __name__ == "__main__":
    raw, work = Path(sys.argv[1]), Path(sys.argv[2])
    figures = sys.argv[3:] or ["pinches", "handles", "ecs"]
    surfs, fills = prepare(raw, work)
    if "pinches" in figures:
        pinches(raw, work)
    if "handles" in figures:
        handles(surfs, fills)
    if "ecs" in figures:
        ecs(surfs, work)
