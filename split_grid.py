"""
Split a MICrONS cube into an n x n x n grid of blocks whose caps agree on
every seam plane, every seam line (where two seam planes meet) and every
seam point (where three meet). Generalizes split_blocks.py (one cut in z).
A thin wrapper over emimesh.blocks.joint_surface (labels shifted per block,
interfaces faired with the cut-plane coordinates fixed, caps relaxed in their
planes, one simplification), see there.

    python split_grid.py OUTDIR [--size 1500] [--n 3] [--influence-radius R]

Writes OUTDIR/joint_surf_dec.vtk with field data offset (int), n, cuts
(3 x (n + 1) plane coordinates per axis, the outer box included).
"""

import argparse
import os

from imagemesh.image import np2pv

from emimesh.blocks import joint_surface
from emimesh.download_data import download_cloudvolume
from emimesh.process_image_data import process_image

parser = argparse.ArgumentParser()
parser.add_argument("outdir")
parser.add_argument("--size", type=float, default=1500.0, help="cube edge (nm)")
parser.add_argument("--n", type=int, default=3, help="blocks per axis")
parser.add_argument("--dx", type=float, default=20.0)
parser.add_argument("--edge", type=float, default=75.0, help="max edge length (L, nm)")
parser.add_argument(
    "--influence-radius", type=int, default=None, help="fair: range of influence (voxels)"
)
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

surf = joint_surface(
    img,
    args.n,
    simplify_eps=0.05 * dx,
    max_edge_length=args.edge,
    influence_radius=args.influence_radius,
)
print(f"joint surface: {surf.n_points} points, cuts {surf.field_data['cuts'].tolist()}")
surf.save(os.path.join(args.outdir, "joint_surf_dec.vtk"))
