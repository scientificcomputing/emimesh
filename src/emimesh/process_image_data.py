import numpy as np
import argparse
import yaml
import time
import pyvista as pv
from pathlib import Path
from utils import np2pv
from dask_image.ndinterp import affine_transform
import fastremap
import dask.array as da
import dask
from functools import partial
import cc3d
import numba
import nbmorph

dask.config.set({"array.chunk-size": "1024 MiB"})

def mergecells(img, labels):
    print(f"merging cells: {labels},  ({img.shape})")
    img = np.where(np.isin(img, labels), labels[0], img)
    return img

def ncells(img, ncells, keep_cell_labels=None):
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    cell_labels = cell_labels[np.argsort(cell_counts)]
    if keep_cell_labels is None: cois =[]
    cois = set(keep_cell_labels)
    for cid in cell_labels:
        if len(cois) >= ncells: break
        cois.add(cid)
    img = np.where(np.isin(img, cois), img, 0)
    return img
    
def dilate(img, radius, labels=None):
    print(f"dilating cells,  ({img.shape})")
    if labels is None:
        img = nbmorph.dilate_labels_spherical(img, radius=radius)
    else:
        vipimg = np.where(np.isin(img, labels), img, 0)
        vipimg = dilate(vipimg, radius=radius)
        img = np.where(vipimg, vipimg, img)
    return img

def erode(img, radius, labels=None):
    print(f"eroding cells,  ({img.shape})")
    if labels is None:
        img = nbmorph.erode_labels_spherical(img, radius=radius)
    else:
        vipimg = np.where(np.isin(img, labels), img, 0)
        vipimg = erode(vipimg, radius=radius)
        orig_wo_vips = np.where(np.isin(img, labels), 0, img)
        img = np.where(orig_wo_vips > vipimg, orig_wo_vips, vipimg)
    return img

def smooth(img, iterations, radius, labels=None):
    print(f"smoothing cells,  ({img.shape})")
    if labels is None:
        img = nbmorph.smooth_labels_spherical(img, radius=radius,
                                              iterations=iterations, dilate_radius=radius)
    else:
        vipimg = np.where(np.isin(img, labels), img, 0)
        vipimg = smooth(vipimg, iterations=iterations, radius=radius)
        # remove labelled cells from original image
        orig_wo_vips = np.where(np.isin(img, labels), 0, img)
        # insert smoothed labeled cells in original (overwrite original)
        img = np.where(vipimg, vipimg, orig_wo_vips)
    return img

def removeislands(img, minsize):
    return cc3d.dust(img, threshold=minsize, connectivity=6)

opdict ={"merge": mergecells, "smooth":smooth, "dilate":dilate,
         "erode":erode, "removeislands":removeislands, "ncells":ncells}

def _parse_to_dict(values):
    result = {}
    for value in values:
        k, v = value.split('=')
        result[k.strip(" '")] = yaml.safe_load(v.strip(" '"))
    return result

def parse_operations(ops):
    parsed = []
    for op in ops:
        subargs =  _parse_to_dict(op[1:])
        parsed.append((op[0], subargs))
    return parsed
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="input data", type=str)
    parser.add_argument(
        "--output", help="output filename", type=str, default="processeddata.vtk"
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
