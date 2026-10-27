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

# a 1 x 1 x 0.5 µm block of the MICrONS segmentation (position in voxels, size in nm)
url = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
img, res = download_cloudvolume(
    url, mip=1, pos=[162258, 200928, 20530], physical_size=[1000, 1000, 500]
)
raw = np2pv(img, res)

dx = 20
ops = [
    ["smooth", "iterations=1", "radius=2"],
    ["erode", "radius=1"],
    ["removeislands", "minsize=150"],  # fragments too small to mesh
]
img, *_ = process_image(raw, dx=dx, operations=ops)  # cells get the labels 2, 3, ...
imggrid = prepare_labels(img)  # ECS = 1


def sub_surface(surf, keep, blabels):
    """Faces keep of surf with the labels blabels, referencing only their own points."""
    faces = surf.regular_faces[keep]
    used, faces = np.unique(faces, return_inverse=True)
    sub = pv.PolyData.from_regular_faces(surf.points[used], faces.reshape(-1, 3))
    sub.cell_data["boundary_labels"] = blabels
    return sub


def block_surface(surf, top, offset):
    """
    Closed surface of the top (labels >= offset) or bottom block: the faces
    touching a label of the block, with the labels of the other block set to
    0, so the interface on the cut plane becomes the block's cap. Faces are
    reoriented such that the block's label comes first, as in contour_labels.
    """
    bl = surf.cell_data["boundary_labels"]
    in_top = bl >= offset
    in_block = in_top if top else (bl > 0) & ~in_top
    keep = in_block.any(axis=1)
    labels = np.where(in_block, bl - offset * top, 0)[keep]
    sub = sub_surface(surf, keep, labels)
    swap = labels[:, 0] == 0
    faces = sub.regular_faces
    faces[swap] = faces[swap][:, ::-1]
    sub.faces = pv.PolyData.from_regular_faces(sub.points, faces).faces
    sub.cell_data["boundary_labels"][swap] = labels[swap][:, ::-1]
    return sub


nx, ny, nz = np.array(imggrid.dimensions) - 1
dz = imggrid.spacing[2]
k_cut = nz // 2
z_cut = imggrid.origin[2] + k_cut * dz

# 1. Unsmoothed surface net of the whole image, with the labels of the top
#    block (k < k_cut) shifted by offset: every voxel face on the cut plane
#    becomes an interface between the blocks
data = imggrid.cell_data["data"].reshape((nx, ny, nz), order="F").astype(np.int64)
# no interface may lie in the cut plane: its points could not leave the plane
# in the smoothing, leaving a flat step where a surface crosses it. Copying the
# last layer of the top block into the bottom block moves these interfaces
# one voxel down
data[:, :, k_cut] = data[:, :, k_cut - 1]
offset = data.max() + 1
data[:, :, :k_cut] += offset
grid = pv.ImageData(dimensions=imggrid.dimensions, spacing=imggrid.spacing, origin=imggrid.origin)
grid.cell_data["data"] = data.flatten(order="F")
surf = close_quad_holes(
    grid.contour_labels("all", smoothing=False, output_mesh_type="quads", background_value=0)
)

# 2. Smooth like extract_surface, with the stencils of the surface net without
#    the offset: the faces on the cut plane with the same label on both sides
#    (a and a + offset) are no interfaces and must not create junctions; their
#    points are only smoothed within the plane. Points on the box stay on their
#    box face, points on the cut plane in the plane. The displacement is
#    limited to 1.2 * dz (< one voxel), so no vertex can move across it
quads = surf.regular_faces
bl = surf.cell_data["boundary_labels"]
bl = np.where(bl >= offset, bl - offset, bl)
cap = bl[:, 0] == bl[:, 1]
on_interface = np.zeros(surf.n_points, dtype=bool)
on_interface[quads[~cap]] = True
A = sp.diags(on_interface.astype(float)) @ build_stencils(
    quads[~cap], surf.n_points, degenerate=None
)
A += sp.diags((~on_interface).astype(float)) @ build_stencils(
    quads[cap], surf.n_points, degenerate=None
)
fixed = bounding_box_mask(surf.points)
fixed[:, 2] |= np.isclose(surf.points[:, 2], z_cut, rtol=0, atol=1e-3 * dz)
points = constrained_smooth(surf.points, A, distance=1.2 * dz, fixed=fixed)
labels = surf.cell_data["boundary_labels"]
surf = pv.PolyData.from_regular_faces(points, triangulate_quads(points, quads))
surf.cell_data["boundary_labels"] = np.repeat(labels, 2, axis=0)

# 3. Whole domain surface: original labels, faces inside a cell or the ECS dropped
bl = surf.cell_data["boundary_labels"]
bl = np.where(bl >= offset, bl - offset, bl)
keep = bl[:, 0] != bl[:, 1]
sub_surface(surf, keep, bl[keep]).save("full_surf_smooth.vtk")

# 4. Closed block surfaces. The cut plane vertices are kept in the
#    simplification, so the caps stay identical
for name, top in (("top", True), ("bottom", False)):
    block = block_surface(surf, top, offset)
    block.save(f"{name}_surf_smooth.vtk")
    on_cut = np.isclose(block.points[:, 2], z_cut, rtol=0, atol=1e-3 * dz)
    block = simplify_surface(block, epsilon=dx * 0.05, fixed=on_cut)
    print(f"{name}: {block.n_points} points after simplification")
    block.save(f"{name}_surf_dec.vtk")
