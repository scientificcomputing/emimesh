"""Tetwild top/bottom blocks as split_blocks.py does; save the meshes and their cut-plane faces."""

import sys

import numpy as np
import pytetwild
import pyvista as pv

from emimesh.winding_number import label_points

name = sys.argv[1]
zc = {"top": 950.0, "bottom": 1050.0}[name]
surf = pv.read(f"../../emimesh/{name}_surf_dec.vtk")
tw = dict(
    stop_energy=10,
    loglevel=5,
    quiet=True,
    disable_filtering=True,
    edge_length_fac=0.05,
    epsilon=1e-3,
    coarsen=False,
    num_threads=16,
)
mesh = pytetwild.tetrahedralize_pv(surf, **tw)
mesh.cell_data["marker"] = label_points(
    surf.points, surf.regular_faces, surf.cell_data["boundary_labels"], mesh.cell_centers().points
)
mesh = mesh.extract_cells(mesh.cell_data["marker"] > 0)
mesh.save(f"{name}_tw.vtu")
# cut-plane boundary faces with the marker of the adjacent tet; the cap stays within
# tetwild's envelope (~2.1 units here) of the plane
bnd = mesh.extract_surface(pass_cellid=True, algorithm="dataset_surface")
bnd.cell_data["tet_id"] = bnd.cell_data["vtkOriginalCellIds"].copy()
z = bnd.points[:, 2][bnd.regular_faces]
cut = bnd.extract_cells((np.abs(z - zc) <= 3.0).all(axis=1)).extract_surface(
    algorithm="dataset_surface"
)
cut.cell_data["marker"] = mesh.cell_data["marker"][cut.cell_data["tet_id"]]
cut.save(f"{name}_cut.vtk")
print(
    f"{name}: tets {mesh.n_cells}, cut tris {cut.n_cells}, max |z-zc| "
    f"{np.abs(cut.points[:, 2] - zc).max():.2f}, markers {np.unique(cut.cell_data['marker']).size}"
)
