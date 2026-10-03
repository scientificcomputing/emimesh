import fastremap
import numpy as np
import pytetwild
import pyvista as pv

from emimesh.download_data import download_cloudvolume
from emimesh.process_image_data import process_image
from emimesh.surface_smoothing import (
    bounding_box_mask,
    build_stencils,
    constrained_smooth,
    triangulate_quads,
)
from emimesh.utils import np2pv
from emimesh.winding_number import label_points
from emimesh.surface_simplification import simplify_surface

path = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
position = (162258, 200928, 20530)
size = (2000,) * 3
dx = 20
ops = [
    ["removeislands", "minsize=5000"],
    ["dilate", "radius=1"],
    ["mode", "iterations=2"],
    ["smooth", "iterations=1", "radius=1"],
    ["upsample", "factor=2"],
    ["mode", "iterations=1"],
    ["zero_edges"],
    ["mode", "iterations=1"],
    ["zero_edges"],
    # ["erode", "radius=1", "struct_sequence='D'"],
    # ["mode", "iterations=1"],
    # ["removeislands", "minsize=5000"],
]
ncells = 100

img, res = download_cloudvolume(path, 1, position, size)
img, remap = fastremap.renumber(img)
imggrid = np2pv(img, res)
imggrid.save("orig.vti")

(imggrid, combinedmap, cell_labels, cell_counts) = process_image(
    imggrid, dx=dx, operations=ops, ncells=ncells, num_threads=1
)

imggrid["data"][imggrid["data"] == 0] = 1


def mark_mesh(mesh, surf):
    """
    Mark each tetrahedron with its anatomical label using the winding number of the input surfaces.

    Parameters:
        mesh: tetrahedralized pyvista.UnstructuredGrid
        surf: multi-label surface mesh with 'boundary_labels' cell data
    """
    marker = label_points(
        surf.points, surf.regular_faces, surf.cell_data["boundary_labels"], mesh.cell_centers().points
    )
    mesh.cell_data["marker"] = marker
    mesh = mesh.extract_cells(marker > 0)
    return mesh


def block_image(img, k_lo, k_hi):
    """Sub-image of the voxel layers k_lo <= k < k_hi (cell data)."""
    nx, ny, nz = np.array(img.dimensions) - 1
    dz = img.spacing[2]
    block = pv.ImageData(
        dimensions=(nx + 1, ny + 1, k_hi - k_lo + 1),
        spacing=img.spacing,
        origin=np.array(img.origin) + (0, 0, k_lo * dz),
    )
    data = img.cell_data["data"].reshape((nx, ny, nz), order="F")[:, :, k_lo:k_hi]
    block.cell_data["data"] = data.flatten(order="F")
    return block


def merge_surfaces(surfs, eps):
    """
    Merge the points of the block surfaces (coincident within eps). Returns the
    merged points and, per block, its quads in merged point ids.
    """
    pts = np.vstack([s.points for s in surfs]).astype(np.float64)
    _, idx, inv = np.unique(np.round(pts / eps), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    offsets = np.cumsum([0] + [s.n_points for s in surfs])
    quads = [inv[s.regular_faces + o] for s, o in zip(surfs, offsets[:-1])]
    return pts[idx], quads


def exterior_faces(quads, blabels):
    """
    Drop the internal cut-plane caps: quads shared by two neighbouring blocks
    (same vertex set, appearing twice in the combined quad list) are not part
    of the whole domain's boundary. Where the labels differ across the cut
    plane, the two caps (a, 0) and (b, 0) are replaced by the interface
    (a, b), with the orientation of the first cap (normal from a to b).
    """
    key = np.sort(quads, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    keep = counts[inv] == 1
    shared = np.flatnonzero(~keep)
    pairs = shared[np.argsort(inv[shared], kind="stable")].reshape(-1, 2)
    assert np.all(blabels[pairs, 1] == 0), "caps must be (label, background)"
    inner = blabels[pairs, 0]
    differ = inner[:, 0] != inner[:, 1]
    keep[pairs[differ, 0]] = True
    blabels = blabels.copy()
    blabels[pairs[differ, 0]] = inner[differ]
    return quads[keep], blabels[keep]


def extract_block(points, quads, blabels):
    """Triangulated block surface referencing only its own points."""
    tris = triangulate_quads(points, quads)
    used, tris = np.unique(tris, return_inverse=True)
    tris = tris.reshape(-1, 3)
    block = pv.PolyData(points[used], np.column_stack([np.full(len(tris), 3), tris]).ravel())
    block.cell_data["boundary_labels"] = np.repeat(blabels, 2, axis=0)
    return block


def extract_plane_boundary(mesh, zc, tol):
    """Boundary faces of the tet mesh with all vertices within tol of the plane z=zc."""
    bnd = mesh.extract_surface()
    z = bnd.points[:, 2][bnd.regular_faces]
    on_plane = (np.abs(z - zc) <= tol).all(axis=1)
    return bnd.extract_cells(on_plane).extract_surface()


nx, ny, nz = np.array(imggrid.dimensions) - 1
gap_start = 95
gap_end = 105

dz = imggrid.spacing[2]
z_min = imggrid.bounds[4]
cut_planes = (z_min + gap_start * dz, z_min + gap_end * dz)
block_k = {"top": (0, gap_start), "mid": (gap_start, gap_end), "bottom": (gap_end, nz)}

# 1. Unsmoothed, closed surface nets of the blocks. Each block is padded with
#    background, so it is closed by a cap on the cut planes; the caps of
#    neighbouring blocks consist of the same quads (one per voxel column)
names = list(block_k)
surfs_raw = [
    block_image(imggrid, *block_k[name]).contour_labels(
        "all", smoothing=False, output_mesh_type="quads", background_value=0
    )
    for name in names
]

# 2. Merge the blocks, so that points on the cut planes are shared and move
#    together. The cut plane quads appear twice; the stencils count them once
eps = 1e-3 * dz
points, quads = merge_surfaces(surfs_raw, eps)
all_quads = np.vstack(quads)

# 3. Smooth like contour_labels, but keep points on the bounding box on their
#    box face (block edges are preserved) and points on the cut planes in
#    their plane. The displacement is limited to smoothing_scale * dz
#    (< one voxel), so no vertex can move across a cut plane
smoothing_scale = 1.2
fixed = bounding_box_mask(points)
fixed[:, 2] |= np.isclose(points[:, 2][:, None], cut_planes, rtol=0, atol=eps).any(axis=1)
A = build_stencils(all_quads, len(points))
points = constrained_smooth(points, A, distance=smoothing_scale * dz, fixed=fixed)

# 4. Split into closed, triangulated block surfaces with conforming interfaces
surfs_smooth = {
    name: extract_block(points, q, s.cell_data["boundary_labels"])
    for name, q, s in zip(names, quads, surfs_raw)
}

# 5. Whole domain smoothed surface (all blocks merged, internal cut-plane
#    caps removed)
all_blabels = np.vstack([s.cell_data["boundary_labels"] for s in surfs_raw])
ext_quads, ext_blabels = exterior_faces(all_quads, all_blabels)
full_surf_smooth = extract_block(points, ext_quads, ext_blabels)
full_surf_smooth.save("full_surf_smooth.vtk")

for name in names:
    surf = surfs_smooth[name]
    surf.save(f"{name}_surf_smooth.vtk")
    print(f"start simplifying {name},#points: {surf.n_points} ")
    surf = simplify_surface(surf, epsilon=dx*0.05)
    print(f"finished simplifying {name},#points: {surf.n_points} ")
    surf.save(f"{name}_surf_dec.vtk")

    twild_defaults = dict(
        stop_energy=10,
        loglevel=5,
        quiet=False,
        disable_filtering=True,
        edge_length_fac=0.05,
        epsilon=1e-3,
        coarsen=False,
        num_threads=6,
    )

    mesh = pytetwild.tetrahedralize_pv(surf, **twild_defaults)
    mesh = mark_mesh(mesh, surf)
    mesh.save(f"{name}_mesh.vtk")

    # outer boundary of the tet mesh on the block's cut planes; tetwild only
    # keeps the input surface within its envelope, epsilon * bbox diagonal
    tol = twild_defaults["epsilon"] * np.linalg.norm(np.ptp(surf.points, axis=0))
    k_lo, k_hi = block_k[name]
    for j, zc in enumerate(cut_planes):
        if np.isclose(zc, z_min + k_lo * dz) or np.isclose(zc, z_min + k_hi * dz):
            cut = extract_plane_boundary(mesh, zc, tol)
            cut.save(f"{name}_cut{j}.vtk")
