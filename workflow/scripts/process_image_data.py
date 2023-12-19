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
import cc3d
import fastmorph
import skimage.morphology as skim


def get_roi_mask(img, roi_cells, radius, iterations, res):
    roi_mask = fastremap.mask_except(img, roi_cells) > 0
    for i in range(iterations):
        roi_mask = fastmorph.spherical_dilate(roi_mask, radius=radius*res[0], parallel=4, anisotropy=res)
    return roi_mask

def remap_labels(img, cells=None):
    img, remapping = fastremap.renumber(img, in_place=True)
    if cells is not None:
        startid = max(remapping.values()) + 1
        fastremap.refit(img, startid + len(cells))
        new_ids = [remapping[cid] for cid in cells]
        remap_dict = {new_ids[i]:startid + i for i in range(len(cells))}
        img = fastremap.remap(img, remap_dict, preserve_missing_labels=True, in_place=True)
        remapping.update({cells[i]:startid + i for i in range(len(cells))})
    return img, remapping

def merge_labels(img, labels):
    print(f"merging cells: {labels}")
    return fastremap.remap(img, {l:labels[0] for l in labels}, in_place=True, preserve_missing_labels=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="input data", type=str)
    parser.add_argument("--ncells", help="number of cells", type=int, default=2)
    parser.add_argument(
        "--smoothiter", help="number of smooting iterations", type=int, default=0
    )
    parser.add_argument(
        "--smoothradius", help="radius of smoothing", type=int, default=0
    )
    parser.add_argument(
        "--expand", help="number of voxels for label expansion", type=int, default=0
    )
    parser.add_argument(
        "--shrink", help="number of voxels for label shrinkage", type=int, default=1
    )
    parser.add_argument(
        "--output", help="output filename", type=str, default="processeddata.xdmf"
    )
    parser.add_argument("--roi", nargs='*',  type=int,
                        help="specify region of interest by cell id")
    parser.add_argument("--roidilate", help="dilate roi", type=str, default="0-0")
    parser.add_argument("--cells", nargs='*', type=int,
                        help="specify cell ids to be included")
    parser.add_argument('--merge', nargs='*', type=int,)
    parser.add_argument("--nworkers", help="number of threads", type=int)
    parser.add_argument("--dx", help="target resolution", type=int, default=None)

    args = parser.parse_args()
    parallel = args.nworkers
    print(f"Using {parallel} workers...")

    start = time.time()
    imggrid = pv.read(args.infile)
    img = imggrid["data"]
    dims = imggrid.dimensions
    resolution = imggrid.spacing

    img, remapping = remap_labels(img, args.cells)

    if args.merge:
        img = merge_labels(img, [remapping[l] for l in args.merge])

    img = img.reshape(dims - np.array([1, 1, 1]), order="F")
    
    dx = max(resolution)
    if args.dx:
        dx = args.dx
        scale = np.diag([dx / r for r in resolution] + [1])

        new_dims = [np.floor(d * r / dx) for d,r in zip(dims, resolution)]
        isotropic_img = affine_transform(img, scale,
                                        output_shape=new_dims,
                                        order=0)
        img = np.array(isotropic_img)
        resolution = [dx]*3
    imggrid = np2pv(img, [dx]*3)
    imggrid["resampled"] = img.flatten(order="F")
    
    if args.roi is not None:
        roi_cells = [remapping[int(l)] for l in args.roi]
        roidilate, rioiter = [int(i) for i in args.roidilate.split("-")]
        roimask = get_roi_mask(img, roi_cells, radius=roidilate, iterations=rioiter, res=resolution)
        extended_roi = fastmorph.spherical_dilate(roimask, radius=100, parallel=parallel, anisotropy=resolution)
        img[extended_roi==0] = 0
    else:
        roimask = None

    load = time.time()
    print(f"load data time: {load - start} s")
    cell_labels, cell_counts = get_cell_frequencies(img)
    print(cell_labels)

    freq_time = time.time()
    print(f"get cell frequency time: {freq_time - load} s")

    mesh_statistics = dict(
        resolution=resolution,
        size=img.shape,
        original_cell_labels=cell_labels,
        original_cell_counts=cell_counts,
    )
    if args.ncells < len(cell_labels):
        cois = cell_labels[-args.ncells :]
    else:
        cois = cell_labels


    print("start processing..")

    img = fastremap.mask_except(img, list(cois))
    imggrid["masked"] = img.flatten(order="F")

    cc3d.dust(img, threshold=100, connectivity=6, in_place=True)
    img = fastmorph.fill_holes(img, remove_enclosed=False)
    from IPython import embed; embed()
    
    for i in range(args.expand):
        img = fastmorph.dilate(img, background_only=False, parallel=parallel)
    #img = expand_labels(img, distance=args.expand)
    imggrid["dilated"] = img.flatten(order="F")

    for i in range(args.smoothiter):
        img = skim.opening(img, footprint=skim.ball(4))
        img = skim.closing(img, footprint=skim.ball(4))
        #img = fastmorph.opening(img, background_only=False,parallel=parallel)
        #img = fastmorph.closing(img, parallel=parallel)

    imggrid["smoothed"] = img.flatten(order="F")

    for i in range(args.shrink):
        img = fastmorph.erode(img, parallel=parallel)

    imggrid["eroded"] = img.flatten(order="F")

    img[roimask==0] = 0
    cc3d.dust(img, threshold=100, connectivity=6, in_place=True)
    imggrid["data"] = img.flatten(order="F")

    for i in range(3):
        roimask = fastmorph.erode(roimask,parallel=parallel)
    imggrid["roimask"] = roimask.flatten(order="F")

    proc_time = time.time()
    print(f"total processing time: {proc_time - freq_time} s")

    resdir = Path(args.output).parent

    resdir.mkdir(parents=True, exist_ok=True)
    imggrid.save(args.output)
    save_time = time.time()
    print(f"saving time: {save_time - proc_time} s")

    cell_labels, cell_counts = get_cell_frequencies(img)
    mesh_statistics["cell_labels"] = cell_labels
    mesh_statistics["cell_counts"] = cell_counts
    mesh_statistics["mapping"] = remapping

    for k, v in mesh_statistics.items():
        mesh_statistics[k] = np.array(v).tolist()

    with open(resdir / "imagestatistic.yml", "w") as mesh_stat_file:
        yaml.dump(mesh_statistics, mesh_stat_file)
