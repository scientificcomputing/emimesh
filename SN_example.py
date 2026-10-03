import pyvista as pv
from emimesh.download_data import download_cloudvolume
from emimesh.process_image_data import process_image
from emimesh.utils import np2pv
import numpy as np
import pytetwild
import igl
import fastremap


path = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
position = (162258, 200928, 20530)
size = (2000,) * 3
dx = 20
ops = [["removeislands", "minsize=5000"], 
       ["dilate", "radius=1"],
       ["mode", "iterations=2"],
        ["smooth", "iterations=1","radius=1"],
       ["upsample", "factor=2"],
        ["mode", "iterations=1"],
        ["zero_edges"],
        ["mode", "iterations=1"],
        ["zero_edges"],
        #["erode", "radius=1", "struct_sequence='D'"],
        #["mode", "iterations=1"],
        #["removeislands", "minsize=5000"],
       ]
ncells = 100

img, res = download_cloudvolume(path, 1, position, size)
img, remap = fastremap.renumber(img)
imggrid = np2pv(img, res)
imggrid.save("orig.vti")

(imggrid, combinedmap,
 cell_labels, cell_counts) = process_image(imggrid, dx=dx,
                                           operations=ops,
                                           ncells=ncells, 
                                           num_threads=1)

imggrid["data"][imggrid["data"]==0] = 1


surf = imggrid.contour_labels("all", smoothing=True, background_value=0)
surf_unsmoothed = imggrid.contour_labels("all", smoothing=False, background_value=0)

def snap_to_bounds(surf, surf_unsmoothed):
    surf = surf.copy()
    bnds = surf_unsmoothed.bounds
    
    # X-Axis: Target column 0
    surf.points[np.isclose(surf_unsmoothed.points[:, 0], bnds.x_min), 0] = bnds.x_min
    surf.points[np.isclose(surf_unsmoothed.points[:, 0], bnds.x_max), 0] = bnds.x_max

    # Y-Axis: Target column 1
    surf.points[np.isclose(surf_unsmoothed.points[:, 1], bnds.y_min), 1] = bnds.y_min
    surf.points[np.isclose(surf_unsmoothed.points[:, 1], bnds.y_max), 1] = bnds.y_max

    # Z-Axis: Target column 2
    surf.points[np.isclose(surf_unsmoothed.points[:, 2], bnds.z_min), 2] = bnds.z_min
    surf.points[np.isclose(surf_unsmoothed.points[:, 2], bnds.z_max), 2] = bnds.z_max
    
    return surf

def mark_mesh(mesh, surf):
    """
    Mark each tetrahedron with its anatomical label using the winding number of the input surfaces.

    Parameters:
        mesh: tetrahedralized pyvista.UnstructuredGrid
        surf: multi-label surface mesh with 'boundary_labels' cell data
    """
    labels = np.unique(surf["boundary_labels"])
    marker = np.zeros(mesh.n_cells, dtype=np.int32)
    query_points = np.array(mesh.cell_centers().points)
    F_global = np.array(surf.faces.reshape(-1, 4)[:, 1:])
    V_global = np.array(surf.points)
    blabels = surf.cell_data["boundary_labels"]

    for i, cid in enumerate(labels):
        if i == 0:
            continue

        mask_out = blabels[:, 0] == cid
        F_out = F_global[mask_out]

        mask_in = blabels[:, 1] == cid
        F_in = F_global[mask_in]

        if len(F_in) > 0:
            F_in_flipped = F_in[:, [0, 2, 1]]
            F_label = np.vstack((F_out, F_in_flipped))
        else:
            F_label = F_out

        fwn = igl.fast_winding_number(V_global, F_label, query_points)
        # |fwn| ≈ 1 inside, ≈ 0 outside; abs covers either orientation convention
        marker = np.where(marker == 0, cid * (np.abs(fwn) > 0.5), marker)

    mesh.cell_data["marker"] = marker
    mesh = mesh.extract_cells(marker > 0)
    return mesh

surf = snap_to_bounds(surf, surf_unsmoothed)
surf.save("surf_smooth.vtk")

img = imggrid["data"].reshape(np.array(imggrid.dimensions) -1 , order="F")
np.save("img.npy", img)

print("start decimate...")
surf_dec = surf.decimate(0.8)
surf_dec.save("surf_dec.vtk")
print("finished decimate.")

twild_defaults = dict(
    stop_energy=10,
    loglevel=5,
    quiet=False,
    disable_filtering=True,
    edge_length_fac=0.05,
    epsilon=1e-3,
    coarsen=False,
    num_threads=6
)

mesh = pytetwild.tetrahedralize_pv(surf_dec, **twild_defaults)
mesh = mark_mesh(mesh, surf)
mesh.save("mesh_marked.vtk")






