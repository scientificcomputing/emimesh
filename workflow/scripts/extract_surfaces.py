import os
import numpy as np
import pyvista as pv
import json
import pyacvd
import argparse
from pathlib import Path
from utils import get_bounding_box

from pyvista.core import _vtk_core as _vtk

from pyvista.core.filters import _get_output, _update_alg
from pyvista.core.utilities.helpers import generate_plane

def clip_closed_surface(surf, normal='x', origin=None, tolerance=1e-06, inplace=False, progress_bar=False):

    plane = generate_plane(normal, origin)
    collection = _vtk.vtkPlaneCollection()
    collection.AddItem(plane)

    alg = _vtk.vtkClipClosedSurface()
    alg.SetGenerateFaces(True)
    alg.SetInputDataObject(surf)
    alg.SetTolerance(tolerance)
    alg.SetClippingPlanes(collection)
    _update_alg(alg, progress_bar, 'Clipping Closed Surface')
    result = _get_output(alg)
    if inplace:
        surf.copy_from(result, deep=False)
    else:
        return result


def clip_closed_box(surf, box):
    box.compute_normals(inplace=True)
    centers = box.cell_centers().points
    for midp, n in zip(centers, box.cell_normals):
        clip_closed_surface(surf, normal=-n, origin=midp, inplace=True)
        
def n_point_target(n, mesh_reduction_factor):
    return 50 + n / mesh_reduction_factor


def extract_surface(mask, grid, mesh_reduction_factor, taubin_smooth_iter):
    mesh = grid.contour([0.5], mask.flatten(order="F"), method="marching_cubes")
    surf = mesh.extract_geometry()
    n_points = surf.number_of_points
    clus = pyacvd.Clustering(surf)
    clus.cluster(n_point_target(n_points, mesh_reduction_factor))
    try:
        surf = clus.create_mesh()
    except:
        print("meshing failed")
        return None
    surf.smooth_taubin(taubin_smooth_iter, inplace=True)
    surf.compute_normals(inplace=True, non_manifold_traversal=False)
    return surf


def extract_cell_meshes(
    img,
    cell_labels,
    resolution,
    mesh_reduction_factor=10,
    taubin_smooth_iter=0,
    write_dir=None,
    roisurf=None,
):
    padded = np.pad(img, 1)
    grid = pv.ImageData(dimensions=padded.shape, spacing=resolution, origin=(0, 0, 0))
    os.system(f"rm -rf {write_dir}/*")
    os.system(f"mkdir -p {write_dir}")
    mesh_boxes = {}
    for obj_id in cell_labels:
        print(obj_id)
        mesh = extract_surface(
            padded == obj_id, grid, mesh_reduction_factor, taubin_smooth_iter
        )
        if mesh is None:
            continue
        p = f"{write_dir}/{obj_id}.ply"
        print(mesh.number_of_points)
        if roisurf is not None:
            p_tight = f"{write_dir}/{obj_id}_wt.ply"
            pv.save_meshio(p_tight, mesh)
            #mesh = mesh.clip_surface(roisurf)
            #mesh = mesh.clip_box(roisurf.bounds, invert=False)
            clip_closed_box(mesh, roisurf)
        print(f"saving : {p}")
        if mesh.number_of_points > 10:
            pv.save_meshio(p, mesh)
            mesh_boxes[obj_id] = get_bounding_box([mesh], 0.0)
    return mesh_boxes

def create_csg_json_tree(surface_files):
    tree = {"operation": "union", "left": surface_files[0], "right": surface_files[1]}
    for sf in surface_files[2:]:
        tree = {"operation": "union", "left": tree, "right": sf}
    return tree

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
    args = parser.parse_args()
    outdir = Path(args.outdir)
    img_grid = pv.read(args.infile)
    dims = img_grid.dimensions
    resolution = img_grid.spacing
    img = img_grid["data"].reshape(dims - np.array([1, 1, 1]), order="F")
    cois = list(np.unique(img))
    cois.remove(0)

    if "roimask" in img_grid.array_names:
        roimask = img_grid["roimask"].reshape(dims - np.array([1, 1, 1]), 
                                              order="F")
        roipadded = np.pad(roimask, 1)
        grid = pv.ImageData(dimensions=roipadded.shape, spacing=resolution,
                            origin=(0, 0, 0))
        roisurf = extract_surface(roipadded, grid, 
                                mesh_reduction_factor=10,
                                taubin_smooth_iter=2)
        roicutsurf = None
    else:
        roisurf = get_bounding_box([img_grid], resolution[0]*5 + img_grid.length*0.002)
        roicutsurf = roisurf

    roi_file = outdir / "roi.ply"

    mboxes = extract_cell_meshes(
        img,
        cois,
        resolution,
        mesh_reduction_factor=10,
        taubin_smooth_iter=2,
        write_dir=outdir,
        roisurf=roicutsurf
    )
    mesh_files = [outdir / f"{cid}.ply" for cid in mboxes.keys()]
    roisurf.save(roi_file)
    csg_tree = create_balanced_csg_json_tree([str(f) for f in mesh_files])
    csg_tree = create_balanced_csg_json_tree([str(roi_file), csg_tree])

    with open(outdir / "csgtree.json", "w") as outfile:
        outfile.write(json.dumps(csg_tree))
