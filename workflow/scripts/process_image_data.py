from dask_image.ndmorph import binary_opening, binary_closing, binary_erosion, binary_dilation
import dask.array as da
import numpy as np
from skimage.segmentation import expand_labels
import skimage.morphology as skim
import argparse
import yaml
import json
import time
import pyvista as pv
from pathlib import Path
from utils import get_cell_frequencies, np2pv
import dask
from dask_image.ndinterp import affine_transform
import fastremap

def binary_smoothing(img, radius, dx, iter=1):
    ball = skim.ball(int(radius / dx))
    for i in range(iter):
        img = binary_opening(img, structure=ball, iterations=iter)
        img = binary_closing(img, structure=ball, iterations=iter)
    return img


def label_smoothing(img, label, radius, dx, background_value=0, iter=1):
    mask = da.isin(img, label)
    img[mask] = background_value
    mask = binary_smoothing(mask, iter=iter, radius=radius, dx=dx)
    img[mask] = label
    return img


def label_erosion(img, label, radius, dx, background_value=0):
    mask = da.isin(img, label)
    ball = skim.ball(int(radius / dx))
    img[mask] = background_value
    mask = binary_erosion(mask, structure=ball)
    img[mask] = label
    return img


def simplify_img(
    img, resolution, cois, expand_voxel_dist, smoothing_iter, smoothing_radius, shrink
):
    dx = max(resolution)
    start = time.time()
    img[np.isin(img, cois) == False] = 0
    if expand_voxel_dist > 0:
        img = expand_labels(img, distance=expand_voxel_dist)
    expand_time = time.time()
    print(f"expansion time: {expand_time - start} s")

    expand_time = time.time()
    img_da = da.from_array(img)
    print((img_da > 0).sum().compute())
    if smoothing_iter > 0:
        for cid in cois:
            img_da = label_smoothing(
                img_da, cid, iter=smoothing_iter, radius=smoothing_radius, dx=dx
            )
    img_da.compute()
    smooth_time = time.time()

    print(f"smooth time: { smooth_time - expand_time} s")

    for cid in cois:
        img_da = label_erosion(
            img_da, cid, background_value=0, radius=dx * shrink, dx=dx
        )
    img_da.compute()
    erosion_time = time.time()

    print(f"erosion time: { erosion_time - smooth_time} s")
    return img_da

def set_boundary_value(img, value, extent=1):
    img[0:extent,:,:] = value
    img[-extent:,:,:] = value
    img[:, 0:extent,:] = value
    img[:, -extent:,:] = value
    img[:,:, 0:extent] = value
    img[:,:, -extent:] = value


def get_roi_mask(img, roi_cells, dilate=5, iterations=4):
    roi_mask = da.isin(img, roi_cells)
    ball = skim.ball(dilate)
    roi_mask = binary_dilation(roi_mask, ball, iterations=iterations)
    return roi_mask


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
    parser.add_argument("--roi", nargs='*', 
                        help="specify region of interest by cell id")
    
    parser.add_argument("--cells", nargs='*', 
                        help="specify cell ids to be included")
    parser.add_argument('--merge', type=str)
    parser.add_argument("--nworkers", help="number of threads", type=int)

    args = parser.parse_args()
    dask.config.set(scheduler="threads", num_workers=args.nworkers)
    print(f"Using {args.nworkers} workers...")

    start = time.time()
    imggrid = pv.read(args.infile)
    dims = imggrid.dimensions
    resolution = imggrid.spacing
    img, remapping = fastremap.renumber(imggrid["data"], in_place=True)
    img = img.reshape(dims - np.array([1, 1, 1]), order="F")
    

    dx = 16
    scale = np.diag([dx / r for r in resolution] + [1])

    new_dims = [int(d * r / dx) for d,r in zip(dims, resolution)]
    print(scale)
    print(new_dims)
    isotropic_img = affine_transform(img, scale,
                                     output_shape=new_dims,
                                     order=0)
    img = np.array(isotropic_img)
    assert img.sum() > 0
    
    if args.merge is not None:
        cells_to_merge = [remapping[int(cid)] for cid in args.merge.split("-")]
        print(f"merging cells: {cells_to_merge}")
        img[np.isin(img, cells_to_merge)] = cells_to_merge[0]

    if args.roi is not None:
        roi_cells = [remapping[int(cid)] for cid in args.roi]
        print(f"using mask of the following cells: {roi_cells}")
        roimask = get_roi_mask(img, roi_cells, dilate=5, iterations=3)
        img[roimask==0] = 0
        set_boundary_value(roimask, 0, extent=5)
        roimask = binary_erosion(roimask, skim.ball(1))
        assert roimask.sum() > 0
    else:
        roimask = None

    np2pv(img, [dx]*3).save("resampledroi.vtk")

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
        #if args.cells is not None:
        cois = cell_labels[-args.ncells :]
    else:
        cois = cell_labels

    da_img = simplify_img(
        img,
        resolution,
        cois,
        args.expand,
        args.smoothiter,
        args.smoothradius,
        args.shrink,
    )
    proc_time = time.time()
    print(f"total processing time: {proc_time - freq_time} s")
    img = np.array(da_img)
    #island_threshold = 64
    #img = skim.remove_small_objects(img, island_threshold)

    imggrid = np2pv(img, [dx]*3,
                     roimask=np.array(roimask) if roimask is not None else None)
    imggrid.save(args.output)
    save_time = time.time()
    print(f"saving time: {save_time - proc_time} s")

    cell_labels, cell_counts = get_cell_frequencies(imggrid["data"])
    mesh_statistics["cell_labels"] = cell_labels
    mesh_statistics["cell_counts"] = cell_counts
    mesh_statistics["mapping"] = remapping

    for k, v in mesh_statistics.items():
        mesh_statistics[k] = np.array(v).tolist()

    with open(Path(args.output).parent / "imagestatistic.yml", "w") as mesh_stat_file:
        yaml.dump(mesh_statistics, mesh_stat_file)
