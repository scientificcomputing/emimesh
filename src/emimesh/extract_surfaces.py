
import numpy as np
import pyvista as pv

import pyacvd

from pathlib import Path
import sys
import shutil
import itertools

from pyvista.core import _vtk_core as _vti
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.utilities.helpers import generate_plane

padded = None
grid = None

def clip_closed_surface(surf, normal='x', origin=None, tolerance=1e-06, inplace=False, progress_bar=False):

    plane = generate_plane(normal, origin)
    collection = _vti.vtiPlaneCollection()
    collection.AddItem(plane)
    alg = _vti.vtiClipClosedSurface()
    alg.SetGenerateFaces(True)
    alg.SetInputDataObject(surf)
    alg.SetTolerance(tolerance)
    alg.SetClippingPlanes(collection)
    _update_alg(alg, progress_bar=False, message='Clipping Closed Surface')
    result = _get_output(alg)
    if inplace:
        surf.copy_from(result, deep=False)
        return None
    else:
        return result


def clip_closed_box(surf, box):
    box.compute_normals(inplace=True)
    centers = box.cell_centers().points
    for midp, n in zip(centers, box.cell_normals):
        clip_closed_surface(surf, normal=-n, origin=midp, inplace=True)

def clean_mesh_nan_points(grid: pv.PolyData):
    """
    Replaces NaN point coordinates in a PyVista grid with the mean of their
    connected neighbors using the efficient `point_neighbors` method.

    Args:
        grid: A PyVista PolyData object that may contain NaN values in its points.

    """
    grid = grid
    points = grid.points

    # Find the indices of points where any coordinate is NaN
    nan_point_indices = np.where(np.isnan(points).any(axis=1))[0]

    if nan_point_indices.size == 0:
        return

    print(f"Found {len(nan_point_indices)} points with NaN coordinates. Cleaning them...")

    # Iterate through each point with NaN coordinates
    for point_idx in nan_point_indices:
        # Directly get the indices of the neighboring points
        neighbor_indices = grid.point_neighbors(point_idx)

        if not neighbor_indices:
            # As a fallback, if the point has no neighbors, replace with origin.
            points[point_idx] = [0, 0, 0]
            continue

        # Get the coordinates of the neighbors
        neighbor_coords = points[neighbor_indices]

        # Filter out any neighbors that are also NaN
        valid_neighbors = neighbor_coords[~np.isnan(neighbor_coords).any(axis=1)]

        if valid_neighbors.shape[0] > 0:
            # Calculate the mean of the valid neighbors and replace the NaN point
            points[point_idx] = np.mean(valid_neighbors, axis=0)
        else:
            # Fallback if all neighbors are also NaN
            points[point_idx] = [0, 0, 0]

def n_point_target(n, mesh_reduction_factor):
    return int(50 + n / mesh_reduction_factor)


def extract_surface(mask, grid, mesh_reduction_factor, taubin_smooth_iter, filename=None):
    mesh = grid.contour([0.5], mask.flatten(order="F"), method="marching_cubes")
    origsurf = mesh.extract_geometry()
    origsurf.clear_data()
    n_points = origsurf.number_of_points
    if n_points < 10: return False
    clus = pyacvd.Clustering(origsurf)
    clus.cluster(n_point_target(n_points, mesh_reduction_factor))
    surf = clus.create_mesh()
    surf.smooth_taubin(n_iter=taubin_smooth_iter, inplace=True)
    clean_mesh_nan_points(surf)
    assert np.isnan(surf.points).any() == False
    if surf.number_of_points > 10 and filename is not None:
        print(f"saving : {filename}")
        pv.save_meshio(filename, surf)
    
    return surf

def extract_surf_id(obj_id, mesh_reduction_factor, taubin_smooth_iter,
            filename):
    surf = extract_surface(padded == obj_id, grid, mesh_reduction_factor,
                    taubin_smooth_iter,filename=filename)
    if surf:
        return obj_id
    else: return None
    
def extract_cell_meshes(
    img,
    cell_labels,
    resolution,
    mesh_reduction_factor=10,
    taubin_smooth_iter=5,
    write_dir=None,
    ncpus=1
):
    global padded
    global grid
    padded = np.pad(img, 1)
    grid = pv.ImageData(dimensions=padded.shape, spacing=resolution, origin=(0, 0, 0))

    write_path = Path(write_dir)
    if write_path.exists():
        shutil.rmtree(write_path) 
    write_path.mkdir(parents=True, exist_ok=True)

    args = [(obj_id, mesh_reduction_factor, taubin_smooth_iter, 
                 f"{write_dir}/{obj_id}.ply") for obj_id in cell_labels]

    if sys.platform != "win32":
        import multiprocessing
        multiprocessing.set_start_method("fork")
        with multiprocessing.Pool(ncpus) as pool:  
            surfaces = pool.starmap(extract_surf_id, args)
            pool.close()
            pool.join()
    else:
        surfaces = list(itertools.starmap(extract_surf_id, args))
    return surfaces

def create_balanced_csg_tree(surface_files):
    n = len(surface_files)
    if  n >= 2:
        return {"operation": "union", "left": create_balanced_csg_tree(surface_files[:int(n/2)])
                                    , "right": create_balanced_csg_tree(surface_files[int(n/2):])}
    return surface_files[0]

