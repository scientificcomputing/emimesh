"""
separate_cells on two examples: two crossing one-voxel processes (top), and
neurites of the FlyEM male CNS in the fan-shaped body (16 nm voxels, bottom),
processed with separate_labels first, whose remaining contacts separate_cells
removes. Per row: the input with its contact voxels, the free voxels of the box
program, and the solution.

    python images/plot_separation.py images/separation.png [workdir]

The downloaded and processed image is cached in workdir (default:
results/separation_figure).
"""

import io
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imagemesh.image import np2pv
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from scipy import ndimage as ndi

from emimesh.download_data import download_cloudvolume
from emimesh.process_image_data import process_image
from emimesh.separate import ST26, contact_mask, separate_cells

COL = {1: "#4f8fdf", 2: "#eb6834", 3: "#b8bcc2"}  # third: a large cell, drawn translucent
CONTACT = "#d62728"
GROWN = "#f6c08f"
EDGE = "#00000040"

URL = "precomputed://gs://flyem-male-cns/v1.0/segmentation"
POSITION = [46976, 17536, 26496]  # neuroglancer position, 8 nm voxels
DX = 16
OPS = [
    ["smooth", "iterations=1", "radius=4"],
    ["reconnect", "s=40"],
    ["separate_labels", "side=both", "connectivity=6", "erode_background=True"],
    ["separate_labels"],
    ["prune_tips"],
    ["remove_pinches"],
    ["separate_labels"],
    ["remove_pinches"],
    ["separate_labels", "avoid_pinches=True"],
]
SITE = np.array([376824.0, 140120.0, 213176.0])  # nm, two neurites in contact
HALF = 8  # half width of the close-up (voxels)
WINDOW = 40  # half width of the window separate_cells runs on (voxels)


def crossing(n=13, h=6):
    """Cell 1 runs along x, cell 2 along y one layer above: they touch where they cross."""
    img = np.zeros((n, n, h), np.int64)
    c, z = n // 2, 1
    img[:, c, z] = 1
    img[c, :, z + 1] = 2
    return img


def processed_image(work):
    path = work / "processed.vti"
    if path.exists():
        return pv.read(path)
    img, res, origin = download_cloudvolume(
        URL, 1, POSITION, [4064] * 3, position_resolution=[8, 8, 8], return_origin=True
    )
    grid = np2pv(np.asarray(img).squeeze(), res)
    img, *_ = process_image(grid, dx=DX, operations=OPS, num_threads=16)
    img.origin = origin
    work.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return img


def neurites(work):
    """The close-up of the FB site before and after separate_cells, with the cells
    renumbered: 1, 2 the two smallest cells that change, 3 the largest one."""
    img = processed_image(work)
    lab = np.asarray(img.cell_data["data"]).reshape(np.array(img.dimensions) - 1, order="F")
    c = np.rint((SITE - np.asarray(img.origin)) / DX - 0.5).astype(int)
    lo = np.maximum(c - WINDOW, 0)
    win = np.ascontiguousarray(lab[tuple(slice(a, b + WINDOW + 1) for a, b in zip(lo, c))])
    sep = separate_cells(win, origin=tuple(lo), verbose=False)
    sl = tuple(slice(b - a - HALF, b - a + HALF) for a, b in zip(lo, c))
    a, b = win[sl], sep[sl]
    changed = a != b
    labs = (set(np.unique(a[changed]).tolist()) | set(np.unique(b[changed]).tolist())) - {0}
    labs = sorted(labs, key=lambda v: (a == v).sum())
    ra, rb = np.zeros_like(a), np.zeros_like(b)
    for i, v in enumerate(labs[:2] + labs[2:][-1:]):
        ra[a == v], rb[b == v] = i + 1, i + 1
    return ra, rb


def panel(img, paint=(), wire=None, wire_col="#808080"):
    """Voxel rendering of img (cells in COL, cell 3 translucent; paint: (mask, colour)
    overrides; wire: voxels drawn as edges only), cropped to its content."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    filled = img > 0
    face = np.zeros(img.shape + (4,))
    edge = np.zeros(img.shape + (4,))
    edge[...] = to_rgba(EDGE)
    for lab, col in COL.items():
        face[img == lab] = to_rgba(col, 0.12 if lab == 3 else 1.0)
        if lab == 3:
            edge[img == lab] = to_rgba(EDGE, 0.08)
    for mask, col in paint:
        face[mask & filled] = to_rgba(col)
    ax.voxels(filled, facecolors=face, edgecolors=edge, linewidth=0.4)
    if wire is not None:  # a separate call, so the faces next to cell voxels are drawn
        w = wire & ~filled
        ax.voxels(w, facecolors=(1, 1, 1, 0), edgecolors=to_rgba(wire_col, 0.5), linewidth=0.4)
    ax.set_proj_type("ortho")
    for setlim, n in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), img.shape):
        setlim(0, n)
    ax.set_aspect("equal")
    ax.view_init(elev=18, azim=-55)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, format="png")
    plt.close(fig)
    buf.seek(0)
    im = plt.imread(buf)
    ink = (im[..., :3] < 0.98).any(-1)
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    return im[rows[0] - 4 : rows[-1] + 5, cols[0] - 4 : cols[-1] + 5]


def row(before, after):
    """Input, program and solution panels with their titles."""
    touch = contact_mask(before)
    free = ndi.binary_dilation(touch, ST26, iterations=2)
    grown = (after != before) & (after > 0)
    emptied = (after != before) & (after == 0)
    return [
        (panel(before, [(touch, CONTACT)]), f"input: {int(touch.sum())} contact voxels"),
        (
            panel(before, [(touch, CONTACT)], wire=free & (before != 3)),
            f"program: {int(free.sum())} free voxels",
        ),
        (
            panel(after, [(grown, GROWN)], wire=emptied, wire_col=CONTACT),
            f"solution: {int(grown.sum())} grown, {int(emptied.sum())} emptied",
        ),
    ]


def main(out, work):
    img = crossing()
    rows = [("two crossing one-voxel processes", row(img, separate_cells(img, verbose=False)))]
    ra, rb = neurites(Path(work))
    rows.append(("neurites in the fan-shaped body (16 nm)", row(ra, rb)))

    # row heights from the panels' aspect, so the flat top row takes little height
    hr = [max(im.shape[0] / im.shape[1] for im, _ in panels) for _, panels in rows]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 4 * sum(hr) + 1.2),
        gridspec_kw=dict(height_ratios=hr, hspace=0.12, wspace=0.02),
    )
    for r, (label, panels) in enumerate(rows):
        for c, (im, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
        axes[r, 0].text(-0.04, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=10)  # fmt: skip
    fig.legend(
        handles=[
            Patch(color=COL[1], label="cell"),
            Patch(color=COL[2], label="other cell"),
            Patch(color=COL[3], alpha=0.4, label="third cell"),
            Patch(color=CONTACT, label="contact voxel"),
            Patch(facecolor="none", edgecolor="#808080", label="free voxel"),
            Patch(facecolor="none", edgecolor=CONTACT, label="emptied"),
            Patch(color=GROWN, label="grown into ECS"),
        ],
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.04, right=0.995, top=0.96, bottom=0.05)
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "images/separation.png",
        sys.argv[2] if len(sys.argv) > 2 else "results/separation_figure",
    )
