"""Overview of the topology repair of the processing: reconnect restoring a
cut neck, then one row per rule of remove_pinches (imagemesh), before and
after. python images/plot_topology_repair.py images/topology_repair.png"""

import contextlib
import io
import sys

import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imagemesh.pinches import count_pinches, remove_pinches
from imagemesh.reconnect import reconnect
from matplotlib.patches import Patch
from nbmorph import smooth_labels_spherical

pv.OFF_SCREEN = True
COL = {1: "#4f8fdf", 2: "#eb6834", 3: "#1baf7a"}
CHANGED = "#d62728"


def extrude(mask2d, z0, z1, n=8):
    img = np.zeros((mask2d.shape[0], mask2d.shape[1], n), np.uint8)
    img[:, :, z0:z1] = mask2d[:, :, None]
    return img


def cases():
    top = (200, 85)
    # reconnect: smoothing cuts a thin neck; reconnect restores it
    d = np.zeros((12, 12, 30), np.uint8)
    zz, yy, xx = np.ogrid[:12, :12, :30]
    d[(zz - 6) ** 2 + (yy - 6) ** 2 + (xx - 6) ** 2 < 25] = 1
    d[(zz - 6) ** 2 + (yy - 6) ** 2 + (xx - 23) ** 2 < 25] = 1
    d[5:8, 5:8, 6:23] = 1
    cut = smooth_labels_spherical(d, 2)
    yield (
        "reconnect: cut neck",
        "smoothing cut the neck (grey: before);\nthe cheapest path is restored,\nthickened by t=2 through removed voxels\n(drawn cut open along the neck)",
        "reconnect",
        cut,
        d,
        (200, 30),
    )
    # 1 separate
    m = np.zeros((8, 8), np.uint8)
    for x in range(1, 7):
        for y in range(1, 7):
            m[x, y] = 1 if x + y < 7 else (0 if x + y == 7 else 2)
    yield (
        "rule 1: separate\n(thin ECS layer between cells)",
        "a cell voxel becomes ECS,\nif that cannot split the cell",
        "pinch",
        extrude(m, 1, 3, 4),
        None,
        (200, 40),
    )
    p = np.zeros((9, 9, 5), np.uint8)
    p[1:4, 1:4, 1:4] = 1
    p[4:7, 4:7, 1:4] = 1
    yield (
        "rule 1: separate\n(a cell touching itself at an edge)",
        "a cell voxel becomes ECS,\nif that cannot split the cell",
        "pinch",
        p,
        None,
        top,
    )
    # 2-4: thin rods, where taking any voxel would split or shorten them
    r = np.zeros((7, 8, 8), np.uint8)
    r[0:7, 3, 3] = 1
    r[0:7, 4, 4] = 1
    yield (
        "rule 2: grow\n(two one-voxel rods touching)",
        "separating would split a rod:\ngrow into ECS away from other cells",
        "pinch",
        r,
        None,
        ("yz", 25, 20),
    )
    x = np.zeros((7, 9, 9), np.uint8)
    x[0:7, 1:4, 5:8] = 2
    x[0:7, 5:8, 1:4] = 2
    x[0:7, 4, 4] = 1
    x[0:7, 3, 3] = 0
    x = np.zeros((7, 9, 9), np.uint8)
    x[0:7, 0:4, 4:8] = 2
    x[0:7, 4:8, 0:4] = 2
    x[0:7, 3, 3] = 1
    x[0:7, 4, 4] = 1
    yield (
        "rule 3: exchange\n(rods between two blocks of another cell)",
        "no ECS in the cube: a voxel\nof the other cell joins the rods",
        "pinch",
        x,
        None,
        ("yz", 25, 20),
    )
    w = r.copy()
    w[0:7, 0:2, 5:8] = 2
    w[0:7, 5:8, 0:2] = 3
    yield (
        "rule 4: grow next to a cell\n(rods, other cells next to the gap)",
        "nothing else is possible:\ngrow next to another cell",
        "pinch",
        w,
        None,
        ("yz", 25, 20),
    )


def add_cells(p, img, mask, **kw):
    grid = pv.ImageData(dimensions=np.array(img.shape) + 1)
    grid.cell_data["v"] = np.where(mask, img, 0).ravel(order="F")
    for l in np.unique(img[mask]):
        if l == 0:
            continue
        p.add_mesh(
            grid.threshold([l - 0.5, l + 0.5], scalars="v"),
            color=COL[int(l)],
            show_edges=True,
            **kw,
        )


def add_box(p, mask, **kw):
    if not mask.any():
        return
    grid = pv.ImageData(dimensions=np.array(mask.shape) + 1)
    grid.cell_data["v"] = mask.ravel(order="F").astype(np.uint8)
    p.add_mesh(grid.threshold(0.5, scalars="v"), **kw)


def render(before, after, ref, view, which, opacity=1.0, crop=None):
    if crop is not None:
        before, after = before[crop], after[crop]
        ref = None if ref is None else ref[crop]
    p = pv.Plotter(off_screen=True, window_size=(520, 440))
    p.set_background("white")
    if which == "before":
        add_cells(p, before, before != 0, opacity=opacity)
        if ref is not None:
            add_box(p, (ref != 0) & (before == 0), color="#888888", style="wireframe", line_width=2)
    else:
        add_cells(p, after, (after != 0) & (after == before), opacity=opacity)
        add_cells(p, after, (after != 0) & (after != before), edge_color=CHANGED, line_width=5)
        add_box(
            p,
            (before != 0) & (after == 0),
            color=CHANGED,
            opacity=0.3,
            show_edges=True,
            edge_color=CHANGED,
        )
    if isinstance(view[0], str):  # a named view, then rotated a little
        p.camera_position = view[0]
        p.camera.azimuth = view[1]
        p.camera.elevation = view[2]
    else:
        p.camera_position = "iso"
        p.camera.azimuth = view[0]
        p.camera.elevation = view[1] - 35
    p.reset_camera()
    p.camera.zoom(1.15)
    im = p.screenshot(return_img=True)
    p.close()
    return im


rows = list(cases())
fig, axes = plt.subplots(
    len(rows), 3, figsize=(11, 3.0 * len(rows) + 1.2), gridspec_kw={"width_ratios": [1.05, 1, 1]}
)
for r, (name, rule, what, img, ref, view) in enumerate(rows):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        after = reconnect(ref, img, s=20, t=2) if what == "reconnect" else remove_pinches(img)
    cat = what
    g = int(((img == 0) & (after != 0)).sum())
    s = int(((img != 0) & (after == 0)).sum())
    x = int(((img != 0) & (after != 0) & (img != after)).sum())
    print(
        name.replace("\n", " "),
        "|",
        cat,
        "| ECS->cell",
        g,
        "cell->ECS",
        s,
        "cell->cell",
        x,
        "| pinches",
        count_pinches(img),
        "->",
        count_pinches(after),
    )
    ax = axes[r, 0]
    ax.axis("off")
    ax.text(0.0, 0.62, name, fontsize=11.5, weight="bold", va="center", transform=ax.transAxes)
    ax.text(0.0, 0.25, rule, fontsize=10, va="center", color="#333333", transform=ax.transAxes)
    for c, which in ((1, "before"), (2, "after")):
        axes[r, c].imshow(
            render(
                img,
                after,
                ref,
                view,
                which,
                1.0,
                (slice(None), slice(0, 7), slice(None)) if r == 0 else None,
            )
        )
        axes[r, c].axis("off")
    axes[r, 1].set_title(
        f"before: {count_pinches(img)} pinched cube" + "s" * (count_pinches(img) != 1), fontsize=9.5
    )
    parts = [f"ECS→cell {g}"] * bool(g) + [f"cell→ECS {s}"] * bool(s) + [f"cell→cell {x}"] * bool(x)
    pl = lambda n: f"{n} pinched"
    axes[r, 2].set_title(f"after: {', '.join(parts)}; {pl(count_pinches(after))}", fontsize=9.5)
legend = [
    Patch(facecolor=COL[1], edgecolor="k", label="cell 1"),
    Patch(facecolor=COL[2], edgecolor="k", label="cell 2"),
    Patch(facecolor=COL[3], edgecolor="k", label="cell 3"),
    Patch(facecolor="white", edgecolor="#bbbbbb", label="ECS (label 0): empty space, not drawn"),
    Patch(
        facecolor="white",
        edgecolor=CHANGED,
        linewidth=2.5,
        label="changed voxel (ECS→cell or cell→cell): red edges",
    ),
    Patch(
        facecolor=CHANGED, alpha=0.3, edgecolor=CHANGED, label="voxel turned into ECS (cell→ECS)"
    ),
    Patch(
        facecolor="white",
        edgecolor="#888888",
        linestyle="--",
        label="cell before the operation only (grey wireframe)",
    ),
]
H = 3.0 * len(rows) + 1.2
fig.suptitle(
    "Topology repair: reconnect (s=20, t=2), then remove_pinches (rules in order of preference)",
    y=1 - 0.15 / H,
    fontsize=12.5,
    weight="bold",
)
fig.legend(
    handles=legend,
    loc="upper center",
    ncol=3,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, 1 - 0.45 / H),
)
plt.tight_layout(rect=(0, 0, 1, 1 - 1.25 / H))
plt.savefig(sys.argv[1], dpi=110)
print("saved")
