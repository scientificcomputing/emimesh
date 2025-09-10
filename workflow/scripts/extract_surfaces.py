import os
import numpy as np
import pyvista as pv
import json
import pyacvd
import argparse
from pathlib import Path
from utils import get_bounding_box
import fastremap

from pyvista.core import _vtk_core as _vtk
from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.utilities.helpers import generate_plane

padded = None
grid = None

def clip_closed_surface(surf, normal='x', origin=None, tolerance=1e-06, inplace=False, progress_bar=False):

    plane = generate_plane(normal, origin)
    collection = _vtk.vtkPlaneCollection()
    collection.AddItem(plane)
    alg = _vtk.vtkClipClosedSurface()
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
        
def n_point_target(n, mesh_reduction_factor):
    return int(50 + n / mesh_reduction_factor)


def extract_surface(mask, grid, mesh_reduction_factor, taubin_smooth_iter, filename=None):
    mesh = grid.contour([0.5], mask.flatten(order="F"), method="marching_cubes")
    surf = mesh.extract_geometry()
    surf.clear_data()
    n_points = surf.number_of_points
    clus = pyacvd.Clustering(surf)
    clus.cluster(n_point_target(n_points, mesh_reduction_factor))
    surf = clus.create_mesh()
    surf.smooth_taubin(n_iter=taubin_smooth_iter, inplace=True)
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
    taubin_smooth_iter=0,
    write_dir=None,
    ncpus=1
):
    global padded
    global grid
    padded = np.pad(img, 1)
    grid = pv.ImageData(dimensions=padded.shape, spacing=resolution, origin=(0, 0, 0))
    os.system(f"rm -rf {write_dir}/*")
    os.system(f"mkdir -p {write_dir}")
    import multiprocessing
    from multiprocessing import Pool
    multiprocessing.set_start_method("fork")
    with Pool(ncpus) as pool:
        args = [(obj_id, mesh_reduction_factor, taubin_smooth_iter, 
                 f"{write_dir}/{obj_id}.ply") for obj_id in cell_labels]
        surfaces = pool.starmap(extract_surf_id, args)
        pool.close()
        pool.join()
    return surfaces

def create_balanced_csg_json_tree(surface_files):
    n = len(surface_files)
    if  n >= 2:
        return {"operation": "union", "left": create_balanced_csg_json_tree(surface_files[:int(n/2)])
                                    , "right": create_balanced_csg_json_tree(surface_files[int(n/2):])}
    return surface_files[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        help="path to image data",
        type=str,
    )
    parser.add_argument(
        "--outdir", help="directory for output", type=str, default="output"
    )
    parser.add_argument(
        "--ncpus", help="number oc cores, default 1", type=int, default=1
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    print(f"reading file: {args.infile}")
    img_grid = pv.read(args.infile)
    dims = img_grid.dimensions
    resolution = img_grid.spacing
    img = img_grid["data"].reshape(dims - np.array([1, 1, 1]), order="F")
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    cell_labels = list(cell_labels[np.argsort(cell_counts)])
    cell_labels.remove(0)

    outerbox = get_bounding_box(img_grid, resolution[0]*5 + img_grid.length*0.002)

    if "roimask" in img_grid.array_names:
        roimask = img_grid["roimask"].reshape(dims - np.array([1, 1, 1]), 
                                              order="F")
        roipadded = np.pad(roimask, 1)
        grid = pv.ImageData(dimensions=roipadded.shape, spacing=resolution,
                            origin=(0, 0, 0))
        roisurf = extract_surface(roipadded, grid, mesh_reduction_factor=10,
                                  taubin_smooth_iter=5)
        
        clip_closed_box(roisurf, outerbox)
    else:
        roisurf = outerbox
    roi_file = outdir / "roi.ply"

    surfs = extract_cell_meshes(
        img,
        cell_labels,
        resolution,
        mesh_reduction_factor=10,
        taubin_smooth_iter=5,
        write_dir=outdir,
        ncpus=args.ncpus
    )

    mesh_files = [outdir / f"{cid}.ply" for cid in surfs]
    roisurf.save(roi_file)
    csg_tree = create_balanced_csg_json_tree([str(f) for f in mesh_files])
    csg_tree = create_balanced_csg_json_tree([str(roi_file), csg_tree])

    with open(outdir / "csgtree.json", "w") as outfile:
        outfile.write(json.dumps(csg_tree))
