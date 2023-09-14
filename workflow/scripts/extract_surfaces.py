import os
import numpy as np
import pyvista as pv
import json
import pyacvd
import argparse
from pathlib import Path
from utils import get_bounding_box


def extract_surface(mask, grid, mesh_reduction_factor, taubin_smooth_iter):
    mesh = grid.contour([0.5], mask.flatten(order="F"), method="marching_cubes")
    surf = mesh.extract_geometry()
    n_points = surf.number_of_points
    if n_points < 50:
        return None
    clus = pyacvd.Clustering(surf)
    clus.cluster(int(n_points / mesh_reduction_factor))
    surf = clus.create_mesh()
    surf.smooth_taubin(taubin_smooth_iter, inplace=True)
    surf.compute_normals(inplace=True, non_manifold_traversal=False)
    return surf


def extract_cell_meshes_pv(
    img,
    cell_labels,
    resolution,
    mesh_reduction_factor,
    taubin_smooth_iter,
    write_dir=None,
):
    padded = np.pad(img, 1)
    grid = pv.ImageData(dimensions=padded.shape, spacing=resolution, origin=(0, 0, 0))
    os.system(f"rm -rf {write_dir}/*")
    os.system(f"mkdir -p {write_dir}")
    mesh_boxes = []
    for obj_id in cell_labels:
        print(obj_id)
        mesh = extract_surface(
            padded == obj_id, grid, mesh_reduction_factor, taubin_smooth_iter
        )
        if mesh is None:
            continue
        p = f"{write_dir}/{obj_id}.ply"
        mesh.save(p)
        mesh_boxes.append(get_bounding_box([mesh], 0.0))
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
        write_dir=outdir,
        mesh_reduction_factor=2,
        taubin_smooth_iter=1,
    )
    mesh_files = [outdir / f"{cid}.ply" for cid in cois]
    bbox_file = outdir / "bbox.ply"
    bbox = get_bounding_box(mboxes)
    bbox.save(bbox_file)
    csg_tree = create_csg_json_tree([str(f) for f in [bbox_file] + mesh_files])

    with open(outdir / "csgtree.json", "w") as outfile:
        outfile.write(json.dumps(csg_tree))
