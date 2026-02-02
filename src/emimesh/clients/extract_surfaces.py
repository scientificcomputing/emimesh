import argparse
import pathlib as Path
import pyvista as pv

from emimesh.extract_surfaces import extract_surface, clip_closed_box, extract_cell_meshes,create_balanced_csg_tree
import numpy as np
import json
import fastremap
from emimesh.utils import get_bounding_box
import argparse

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
        cell_labels[::-1],
        resolution,
        mesh_reduction_factor=10,
        taubin_smooth_iter=5,
        write_dir=outdir,
        ncpus=args.ncpus
    )

    mesh_files = [outdir / f"{cid}.ply" for cid in surfs if cid]
    roisurf.save(roi_file)
    csg_tree = create_balanced_csg_tree([str(f) for f in mesh_files])
    csg_tree = create_balanced_csg_tree([str(roi_file), csg_tree])
    csg_tree = {"operation":"intersection","right":csg_tree, "left":str(roi_file)}
    with open(outdir / "csgtree.json", "w") as outfile:
        outfile.write(json.dumps(csg_tree))
