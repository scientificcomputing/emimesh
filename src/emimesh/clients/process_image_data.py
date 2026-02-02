import argparse
import time
import pyvista as pv
from pathlib import Path
from emimesh.utils import np2pv
from dask_image.ndinterp import affine_transform
import dask.array as da
from functools import partial
import numba
import numpy as np
import fastremap
import dask
import yaml
from emimesh.process_image_data import opdict, parse_operations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="input data", type=str)
    parser.add_argument(
        "--output", help="output filename", type=str, default="processeddata.vti"
    )
    parser.add_argument("--nworkers", help="number of workers", type=int, default=1)
    parser.add_argument("--dx", help="target resolution", type=int, default=None)
    parser.add_argument("--ncells", help="number of cells", type=int, default=None)
    parser.add_argument('-o','--operation', nargs='+', action='append', help="operations to be performed on the segmented image")

    args = parser.parse_args()
    n_parallel = args.nworkers
    numba.set_num_threads(n_parallel)
    print(f"Using {n_parallel} workers...")
    start = time.time()

    # read image file
    imggrid = pv.read(args.infile)
    img = imggrid["data"]
    dims = imggrid.dimensions
    resolution = imggrid.spacing
    img = img.reshape(dims - np.array([1, 1, 1]), order="F")
    img = da.from_array(img)

    # get cells labels, and filter by n-largest (if requested)
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    if args.ncells:
        cell_labels = list(cell_labels[np.argsort(cell_counts)])
        cell_labels.remove(0)
        cois = list(cell_labels[-args.ncells :])
        img = da.where(da.isin(img, cois), img, 0)
    else:
        cell_labels = list(cell_labels)
        if 0 in cell_labels: cell_labels.remove(0)

    # remap labels to smaller, sequential ints
    remapping = {int(c):i for i,c in enumerate([0] + cell_labels)}
    remap = lambda ids: [remapping[int(i)] for i in ids if i in remapping.keys()]
    img = img.map_blocks(partial(fastremap.remap, table=remapping), dtype=img.dtype)
    img = img.map_blocks(partial(fastremap.refit, value=len(cell_labels)))

    # interpolate into the specified, isotropic grid with size dx
    dx = args.dx
    scale = np.diag([dx / r for r in resolution] + [1])
    new_dims = [int(d * r / dx) for d,r in zip(dims, resolution)]
    img = affine_transform(img, scale, output_shape=new_dims, order=0,
                           output_chunks=500)
    img = dask.compute(img, num_workers= args.nworkers)[0]
    print(f"image size: {img.shape}")
    resolution = [dx]*3
    roi = None

    # parse user specified operations, and iterate over them:
    operations = parse_operations(args.operation)
    for op, kwargs in operations:
        print(op, kwargs)
        for k in kwargs.keys():
            if "label" in k:
                labels = kwargs[k]
                if kwargs.get("allexcept", False):
                    kwargs[k] = list(set(remapping.values()) - set(remap(labels)))
                else:
                    kwargs[k] = remap(labels)

        if op=="roigenerate":
            roi = np.isin(img, kwargs["labels"])
            continue
        if op=="roiapply":
            img = np.where(roi, img, 0) 
            continue
        if op.startswith("roi"):
            roiop = op[3:]
            roi = opdict[roiop](roi, **kwargs)
        else:
            img =opdict[op](img, **kwargs)
    
    print(f"processed! {img.shape}")
    proc_time = time.time()

    # remap labels to smaller, sequential ints again, since many labels might have disappeared...
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    cell_labels = list(cell_labels[np.argsort(cell_counts)])
    cell_labels.remove(0)

    img = da.array(img)
    remapping2 = {0:0}
    remapping2.update({int(c):i + 2 for i,c in enumerate(cell_labels)})
    img = img.map_blocks(partial(fastremap.remap, table=remapping2), dtype=img.dtype)
    img = img.map_blocks(partial(fastremap.refit, value=max(remapping2.values())))
    img = dask.compute(img, num_workers= args.nworkers)[0]

    imggrid = np2pv(img, resolution)
    if roi is not None:
        imggrid["roimask"] = np.array(roi).flatten(order="F")

    resdir = Path(args.output).parent

    resdir.mkdir(parents=True, exist_ok=True)
    imggrid.save(args.output)
    save_time = time.time()
    print(f"saving time: {save_time - proc_time} s")


    combinedmap = {k:remapping2[v] for k,v in remapping.items() if v in remapping2.keys()}
    mesh_statistics = dict()
    mesh_statistics["cell_labels"] = cell_labels
    mesh_statistics["cell_counts"] = cell_counts
    mesh_statistics["mapping"] = combinedmap

    for k, v in mesh_statistics.items():
        mesh_statistics[k] = np.array(v).tolist()

    with open(resdir / "imagestatistic.yml", "w") as mesh_stat_file:
        yaml.dump(mesh_statistics, mesh_stat_file)


if __name__ == "__main__":
    main()