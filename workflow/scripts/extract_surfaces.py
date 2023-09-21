import os
import numpy as np
import pyvista as pv
import json
import pyacvd
import argparse
from pathlib import Path
from utils import get_bounding_box


def n_point_target(n, mesh_reduction_factor):
    return 50 + n / mesh_reduction_factor


def extract_surface(mask, grid, mesh_reduction_factor, taubin_smooth_iter):
    mesh = grid.contour([0.5], mask.flatten(order="F"), method="marching_cubes")
    surf = mesh.extract_geometry()
    n_points = surf.number_of_points
    print(n_points)
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


def extract_cell_meshes_pv(
    img,
    cell_labels,
    resolution,
    mesh_reduction_factor=10,
    taubin_smooth_iter=0,
    write_dir=None,
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
        pv.save_meshio(p, mesh)
        mesh_boxes[obj_id] = get_bounding_box([mesh], 0.0)
    return mesh_boxes


def extract_cell_meshes_zmesh(
    img,
    cell_labels,
    resolution,
    max_error=None,
    write_dir=None,
):
    from zmesh import Mesher

    mesher = Mesher(resolution)
    padded = np.pad(img, 1)
    mesher.mesh(padded, close=False)

    os.system(f"rm -rf {write_dir}/*")
    os.system(f"mkdir -p {write_dir}")
    mesh_boxes = {}

    for obj_id in cell_labels:
        print(obj_id)
        mesh = mesher.get(
            obj_id,
            normals=True,
            reduction_factor=1000,
            max_error=max_error,
            voxel_centered=True,
        )
        if mesh.vertices.shape[0] < 10:
            continue
        p = f"{write_dir}/{obj_id}.ply"
        with open(p, "wb") as f:
            f.write(mesh.to_ply())
        mesh_boxes[obj_id] = get_bounding_box([mesh.vertices], 0.0)
    return mesh_boxes


def create_csg_json_tree(surface_files):
    tree = {"operation": "union", "left": surface_files[0], "right": surface_files[1]}
    for sf in surface_files[2:]:
        tree = {"operation": "union", "left": tree, "right": sf}
    return tree


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
    img_grid = None
    cois = list(np.unique(img))
    cois.remove(0)
    mboxes = extract_cell_meshes_pv(
        img,
        cois,
        resolution,
        mesh_reduction_factor=10,
        taubin_smooth_iter=2,
        write_dir=outdir,
    )
    mesh_files = [outdir / f"{cid}.ply" for cid in mboxes.keys()]
    bbox_file = outdir / "bbox.ply"
    bbox = get_bounding_box(mboxes.values())
    bbox.save(bbox_file)
    csg_tree = create_csg_json_tree([str(f) for f in [bbox_file] + mesh_files])

    with open(outdir / "csgtree.json", "w") as outfile:
        outfile.write(json.dumps(csg_tree))
