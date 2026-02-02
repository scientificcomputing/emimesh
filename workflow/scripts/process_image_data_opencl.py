import numpy as np
import argparse
import yaml
import json
import time
import pyvista as pv
from pathlib import Path
from utils import get_cell_frequencies, np2pv
from dask_image.ndinterp import affine_transform
import fastremap
import skimage.morphology as skim
import pyclesperanto_prototype as cle
from cle_patch import opening_labels, closing_labels, erode_labels
import dask.array as da
import dask
from functools import partial
from collections.abc import Iterable
import cc3d
cle.set_wait_for_kernel_finish(True)
dask.config.set({"array.chunk-size": "512 MiB"})

def mergecells(img, labels):
    print(f"merging cells: {labels},  ({img.shape})")
    img = np.where(np.isin(img, labels), labels[0], img)
    return img

def ncells(img, ncells):
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    cell_labels = cell_labels[np.argsort(cell_counts)]
    cois = list(cell_labels[-ncells :])
    img = da.where(da.isin(img, cois), img, 0)
    return img
    
def dilate(img, iterations, radius, labels=None):
    print(f"dilating cells,  ({img.shape})")
    if labels is None:
        for i in range(iterations):
            img = cle.dilate_labels(img, radius=radius)
        img = cle.pull(img)
    else:
        vipimg = np.where(np.isin(img, labels), img, 0)
        vipimg = dilate(vipimg, iterations=iterations, radius=radius)
        img = np.where(vipimg, vipimg, img)
    return img

def erode(img, iterations, radius, labels=None):
    print(f"eroding cells,  ({img.shape})")
    if labels is None:
        for i in range(iterations):
            img = erode_labels(img, radius=radius)
        img = cle.pull(img)
    else:
        vipimg = np.where(np.isin(img, labels), img, 0)
        vipimg = erode(vipimg, iterations=iterations, radius=radius)
        orig_wo_vips = np.where(np.isin(img, labels), 0, img)
        img = np.where(orig_wo_vips > vipimg, orig_wo_vips, vipimg)
    return img

def smooth(img, iterations, radius, labels=None):
    print(f"smoothing cells,  ({img.shape})")
    if labels is None:
        for i in range(iterations):
            img = opening_labels(img, radius=radius)
            img = closing_labels(img, radius=radius)
        img = cle.pull(img)
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

    #would be more performant, but relabels all cells..
    #return cle.exclude_small_labels(img, maximum_size=minsize)

opdict ={"merge": mergecells, "smooth":smooth, "dilate":dilate,
         "erode":erode, "removeislands":removeislands, "ncells":ncells}


def _parse_to_dict(values):
    result = {}
    for value in values:
        k, v = value.split('=')
        result[k.strip()] = yaml.safe_load(v.strip())
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
        "--output", help="output filename", type=str, default="processeddata.vti"
    )
    parser.add_argument("--nworkers", help="number of workers", type=int, default=1)
    parser.add_argument("--dx", help="target resolution", type=int, default=None)
    parser.add_argument("--ncells", help="number of cells", type=int, default=None)
    parser.add_argument('-o','--operation', nargs='+', action='append', help="operations to be performed on the segmented image")

    args = parser.parse_args()
    parallel = args.nworkers
    print(f"Using {parallel} workers...")
    print(f"Available  devices {cle.available_device_names()}")
    cle.select_device("NVIDIA")
    dev = cle.get_device()
    print(f"Running processing on {dev}")
    mem_gb = dev.device.global_mem_size / 1e9
    print(f"device memory detected: {mem_gb:.1f} GB")

    start = time.time()
    imggrid = pv.read(args.infile)
    img = imggrid["data"]
    dims = imggrid.dimensions
    resolution = imggrid.spacing
    img = img.reshape(dims - np.array([1, 1, 1]), order="F")
    img = da.from_array(img)
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    if args.ncells:
        cell_labels = list(cell_labels[np.argsort(cell_counts)])
        cell_labels.remove(0)
        cois = list(cell_labels[-args.ncells :])
        img = da.where(da.isin(img, cois), img, 0)
    else:
        cell_labels = list(cell_labels)
        cell_labels.remove(0)

    remapping = {int(c):i for i,c in enumerate([0] + cell_labels)}
    remap = lambda ids: [remapping[int(i)] for i in ids if i in remapping.keys()]
    img = img.map_blocks(partial(fastremap.remap, table=remapping), dtype=img.dtype)
    img = img.map_blocks(partial(fastremap.refit, value=len(cell_labels)))

    dx = args.dx
    scale = np.diag([dx / r for r in resolution] + [1])
    new_dims = [int(d * r / dx) for d,r in zip(dims, resolution)]
    img = affine_transform(img, scale, output_shape=new_dims, order=0,
                           output_chunks=500)
    print(f"image size: {img.shape}")
    resolution = [dx]*3
    roi = None
    operations = parse_operations(args.operation)

    for op, kwargs in operations:
        if "labels" in kwargs.keys():
            labels = kwargs["labels"]
            if kwargs.get("allexcept", False):
                kwargs["labels"] = list(set(remapping.values()) - set(remap(labels)))
            else:
                kwargs["labels"] = remap(labels)

        if op=="roigenerate":
            roi = da.isin(img, kwargs["labels"])
            continue
        if op=="roiapply":
            img = da.where(roi, img, 0) 
            continue
        if op=="ncells":
            img = ncells(img, **kwargs)
            continue
        if op.startswith("roi"):
            roiop = op[3:]
            roi = roi.map_overlap(partial(opdict[roiop], **kwargs), depth=30, dtype=np.uint8)
        else:
            img = img.map_overlap(partial(opdict[op], **kwargs), depth=30, dtype=img.dtype)
    
    chunk_mem_gb = np.prod(img.chunksize) * np.nbytes[img.dtype] / 1e9
    print(f"chunk size: {chunk_mem_gb} GB")
    max_workers = int(mem_gb / (chunk_mem_gb*3)) # assume max 3x chunk memory is used
    img, roi = dask.compute(img, roi, num_workers=min(max_workers, args.nworkers))
    print(f"processed! {img.shape}")
    proc_time = time.time()

    img = da.array(img)
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    cell_labels = list(cell_labels[np.argsort(cell_counts)])
    cell_labels.remove(0)

    img = da.array(img)
    remapping2 = {0:0}
    remapping2.update({int(c):i + 2 for i,c in enumerate(cell_labels)})
    img = img.map_blocks(partial(fastremap.remap, table=remapping2), dtype=img.dtype)
    img = img.map_blocks(partial(fastremap.refit, value=max(remapping2.values())))
    img = dask.compute(img, num_workers=min(max_workers, args.nworkers))[0]

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
