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
import cc3d
cle.set_wait_for_kernel_finish(True)
dask.config.set({"array.chunk-size": "512 MiB"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="input data", type=str)
    parser.add_argument("--ncells", help="number of cells", type=int, default=2)
    parser.add_argument(
        "--smoothiter", help="number of smoothing iterations", type=int, default=0
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
        "--output", help="output filename", type=str, default="processeddata.vtk"
    )
    parser.add_argument("--roi", nargs='*',  type=int,
                        help="specify region of interest by cell id")
    parser.add_argument("--roidilate", help="dilate roi", type=str, default="0-0")
    parser.add_argument("--cells", nargs='*', type=int,
                        help="specify cell ids to be included")
    parser.add_argument('--merge', nargs='*', type=int,)
    parser.add_argument("--nworkers", help="number of workers", type=int, default=1)
    parser.add_argument("--dx", help="target resolution", type=int, default=None)

    args = parser.parse_args()
    parallel = args.nworkers
    print(f"Using {parallel} workers...")
    print(f"Available  devices {cle.available_device_names()}")    
    dev = cle.get_device()
    print(f"Running processing on {dev}")
    mem_gb = dev.device.global_mem_size / 1e9
    print(f"device memory detected: {mem_gb:.1f} GB")
    print(f"smoothing iterations: {args.smoothiter}")
    print(f"smoothing radius: {args.smoothradius}")

    start = time.time()
    imggrid = pv.read(args.infile)
    img = imggrid["data"]
    dims = imggrid.dimensions
    resolution = imggrid.spacing
    img = img.reshape(dims - np.array([1, 1, 1]), order="F")
    img = da.from_array(img)
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    remapping = {int(c):i for i,c in enumerate(cell_labels)}
    img = img.map_blocks(partial(fastremap.remap, table=remapping), dtype=img.dtype)
    img = img.map_blocks(partial(fastremap.refit, value=len(cell_labels)))

    mesh_statistics = dict(
        resolution=resolution,
        size=img.shape,
        original_cell_labels=cell_labels,
        original_cell_counts=cell_counts,
    )


    if args.merge is not None:
        cells_to_merge = [remapping[int(cid)] for cid in args.merge]
        print(f"merging cells: {cells_to_merge}")
        img = da.where(da.isin(img, cells_to_merge), cells_to_merge[0], img)

    if args.ncells < len(cell_labels):
        cell_labels = cell_labels[np.argsort(cell_counts)]
        cois = list(cell_labels[-args.ncells :])
        if args.cells:
            for c in args.cells:
                if c in cois:
                    cois.remove(c)
            cois = args.cells + cois[- (args.ncells - len(args.cells)):]
    else:
        cois = cell_labels
    cois = [remapping[int(cid)] for cid in cois if int(cid) in remapping.keys()]
    img = da.where(da.isin(img, cois), img, 0)
    
    if args.dx:
        dx = args.dx
    else:
        dx = max(resolution)

    scale = np.diag([dx / r for r in resolution] + [1])
    new_dims = [int(d * r / dx) for d,r in zip(dims, resolution)]
    img = affine_transform(img, scale, output_shape=new_dims, order=0,
                           output_chunks=500)
    resolution = [dx]*3

    if args.roi is not None:
        roi_cells = [remapping[int(cid)] for cid in args.roi]
        print(f"using mask of the following cells: {roi_cells}")
        roidilate, rioiter = [int(i) for i in args.roidilate.split("-")]
        roimask = da.isin(img, roi_cells)
        roidilatefunc = lambda chunk: np.array(cle.dilate_labels(chunk, radius=roidilate))
        for i in range(rioiter):
            roimask = roimask.map_overlap(roidilatefunc, depth=roidilate, dtype=np.uint8)
        extended_roimask = roimask.map_overlap(roidilatefunc, depth=roidilate, dtype=np.uint8)
        img = da.where(extended_roimask, img, 0)
        extended_roimask = None
    else:
        roimask = None

    if args.cells:
        vipcells = [remapping[int(cid)] for cid in args.cells if int(cid) in remapping.keys()]
        vipimg = da.where(da.isin(img, vipcells), img, 0)
        dilate_vips = lambda chunk: np.array(cle.dilate_labels(chunk, radius=5))
        vipimg = vipimg.map_overlap(dilate_vips, depth=10)
        img = da.where(vipimg, vipimg, img)
    
    remove_dust = partial(cc3d.dust, threshold=100, connectivity=6)
    img = img.map_overlap(remove_dust, depth=10)

    def process_img(chunk, smoothradius=0, smoothiter=0, expanditer=0, shrinkiter=0):
        print("Processing image of size", chunk.shape)
        for i in range(expanditer):
            chunk = cle.dilate_labels(chunk, radius=1)

        # see: https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20h_segmentation_post_processing/smooth_labels.html
        for i in range(smoothiter):
            chunk = opening_labels(chunk, radius=smoothradius)
            chunk = closing_labels(chunk, radius=smoothradius)
            #chunk = cle.smooth_labels(chunk, radius=smoothradius)

        for i in range(shrinkiter):
            chunk = erode_labels(chunk, radius=1)

        return np.array(chunk)

    process = partial(process_img, smoothradius=args.smoothradius, smoothiter=args.smoothiter,
                      expanditer=args.expand, shrinkiter=args.shrink, )
    img = img.map_overlap(process, depth=10, dtype=img.dtype)
    if roimask is not None:
        print("applying roi mask")
        roimask = roimask.map_blocks(np.array)
        img = da.where(roimask, img, 0)
        img = img.map_overlap(remove_dust, depth=10)

    chunk_mem_gb = np.prod(img.chunksize) * np.nbytes[img.dtype] / 1e9
    max_workers = int(mem_gb / (chunk_mem_gb*3)) # assume max 3x chunk memory is used
    img = img.compute(num_workers=min(max_workers, args.nworkers))
    print(f"processed! {img.shape}")
    proc_time = time.time()
    imggrid = np2pv(img, resolution)

    if roimask is not None:
        eroderoi = lambda chunk: np.array(cle.erode_labels(chunk, radius=2))
        roimask = roimask.map_overlap(eroderoi, depth=10, dtype=roimask.dtype)
        imggrid["roimask"] = np.array(roimask).flatten(order="F")

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
