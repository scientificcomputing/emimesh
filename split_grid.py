"""
Split a MICrONS cube into an n x n x n grid of blocks whose caps agree on
every seam plane, every seam line (where two seam planes meet) and every
seam point (where three meet). Generalizes split_blocks.py (one cut in z).

The labels of block b = i + n j + n^2 k (i, j, k along x, y, z) are shifted
by b * offset, so every voxel face on a cut plane is an interface between
two blocks, and the whole surface is smoothed and simplified once:

- cap faces: labels (a + b offset, a + c offset) with b != c, on a cut plane;
- seam lines are feature curves between the caps of two planes, seam points
  are where three planes meet: the simplification keeps points on their
  planes (endpoint placement), curve vertices collapse along their curve,
  so the caps stay planar and the lines straight.

    python split_grid.py OUTDIR [--size 1500] [--n 3]

Writes OUTDIR/joint_surf_dec.vtk with field data offset (int), n, cuts
(3 x (n + 1) plane coordinates per axis, the outer box included).
"""

import argparse
import os

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from imagemesh.image import np2pv
from imagemesh.simplification import simplify_surface
from imagemesh.smoothing import (
    bounding_box_mask,
    build_stencils,
    constrained_smooth,
    triangulate_quads,
)
from imagemesh.surface import close_quad_holes

from emimesh.download_data import download_cloudvolume
from emimesh.extract_surfaces import prepare_labels
from emimesh.process_image_data import process_image

parser = argparse.ArgumentParser()
parser.add_argument("outdir")
parser.add_argument("--size", type=float, default=1500.0, help="cube edge (nm)")
parser.add_argument("--n", type=int, default=3, help="blocks per axis")
parser.add_argument("--dx", type=float, default=20.0)
parser.add_argument("--edge", type=float, default=75.0, help="max edge length (L, nm)")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

url = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
s = args.size
img, res = download_cloudvolume(url, mip=1, pos=[162258, 200928, 20530], physical_size=[s, s, s])
raw = np2pv(img, res)
dx = args.dx
ops = [
    ["smooth", "iterations=1", "radius=2"],
    ["erode", "radius=1"],
    ["removeislands", "minsize=150"],
]
img, *_ = process_image(raw, dx=dx, operations=ops)  # cells 2, 3, ...
imggrid = prepare_labels(img)  # ECS = 1

n = args.n
dims = np.array(imggrid.dimensions) - 1  # cells per axis
spacing = np.array(imggrid.spacing)
origin = np.array(imggrid.origin)
# voxel index of each cut (0 and dims included)
kcut = [np.rint(np.arange(n + 1) * dims[a] / n).astype(int) for a in range(3)]
cuts = np.array([origin[a] + kcut[a] * spacing[a] for a in range(3)])
print("cells per axis", dims, "cuts", cuts.tolist())

# 1. No interface may lie in a cut plane (split_blocks.py): the layer after
#    each inner cut copies the layer before it. Then shift the labels per block
data = imggrid.cell_data["data"].reshape(tuple(dims), order="F").astype(np.int64)
for a in range(3):
    for k in kcut[a][1:-1]:
        sl_dst = [slice(None)] * 3
        sl_src = [slice(None)] * 3
        sl_dst[a], sl_src[a] = k, k - 1
        data[tuple(sl_dst)] = data[tuple(sl_src)]
offset = int(data.max()) + 1
blk = np.zeros(tuple(dims), np.int64)
for a in range(3):
    idx = np.searchsorted(kcut[a][1:-1], np.arange(dims[a]), side="right")
    shape = [1, 1, 1]
    shape[a] = dims[a]
    blk = blk + idx.reshape(shape) * n**a
data = data + blk * offset
grid = pv.ImageData(dimensions=imggrid.dimensions, spacing=imggrid.spacing, origin=imggrid.origin)
grid.cell_data["data"] = data.flatten(order="F")
surf = close_quad_holes(
    grid.contour_labels("all", smoothing=False, output_mesh_type="quads", background_value=0)
)

# 2. Smooth as split_blocks.py: cap faces (same base label on both sides) are
#    no interfaces; points on cut planes stay in their planes (on two planes:
#    on the seam line, on three: fixed)
quads = surf.regular_faces
bl = surf.cell_data["boundary_labels"]
base = np.where(bl > 0, (bl - 1) % offset + 1, 0)
cap = (base[:, 0] == base[:, 1]) & (bl[:, 0] != bl[:, 1])
on_interface = np.zeros(surf.n_points, dtype=bool)
on_interface[quads[~cap]] = True
A = sp.diags(on_interface.astype(float)) @ build_stencils(
    quads[~cap], surf.n_points, degenerate=None
)
A += sp.diags((~on_interface).astype(float)) @ build_stencils(
    quads[cap], surf.n_points, degenerate=None
)
fixed = bounding_box_mask(surf.points)
for a in range(3):
    for c in cuts[a][1:-1]:
        fixed[:, a] |= np.isclose(surf.points[:, a], c, rtol=0, atol=1e-3 * dx)
points = constrained_smooth(surf.points, A, distance=1.2 * dx, fixed=fixed)
# the fixed coordinates exactly on their planes
for a in range(3):
    for c in cuts[a]:
        points[np.isclose(points[:, a], c, rtol=0, atol=1e-3 * dx), a] = c
surf = pv.PolyData.from_regular_faces(points, triangulate_quads(points, quads))
surf.cell_data["boundary_labels"] = np.repeat(bl, 2, axis=0)

# 3. Simplify the whole surface once; cap edges at most args.edge (tetopt L)
smooth = surf
surf = simplify_surface(smooth, epsilon=dx * 0.05, max_edge_length=args.edge)
print(f"whole surface: {smooth.n_points} -> {surf.n_points} points after simplification")
surf.field_data["offset"] = [offset]
surf.field_data["n"] = [n]
surf.field_data["cuts"] = cuts.ravel()
surf.save(os.path.join(args.outdir, "joint_surf_dec.vtk"))
