import numpy as np
import yaml
import fastremap
import numba
from pathlib import Path

from imagemesh.handles import fill_handles
from imagemesh.morphology import (
    dilate,
    erode,
    mergecells,
    mode,
    ncells,
    removeislands,
    smooth,
    upsample,
    zero_edges,
)
from imagemesh.pinches import remove_pinches

opdict ={"merge": mergecells, "smooth":smooth, "dilate":dilate,
         "erode":erode, "removeislands":removeislands, "ncells":ncells, "mode":mode,
         "zero_edges":zero_edges, "upsample":upsample, "remove_pinches":remove_pinches,
         "fill_handles":fill_handles}

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


@numba.njit(parallel=True, cache=True)
def _gather(img, ix, iy, iz):
    out = np.empty((ix.size, iy.size, iz.size), dtype=img.dtype)
    for i in numba.prange(ix.size):
        for j in range(iy.size):
            for k in range(iz.size):
                out[i, j, k] = img[ix[i], iy[j], iz[k]]
    return out


def resample_nearest(img, resolution, dx):
    """Nearest-neighbour resampling of img onto an isotropic grid with spacing dx,
    mapping voxel centers to voxel centers (identical to scipy's affine_transform
    with order=0, mode="nearest")."""
    scales = [dx / r for r in resolution]
    new_dims = [max(1, round(n * r / dx)) for n, r in zip(img.shape, resolution)]
    idx = [np.clip(np.floor(s * np.arange(n) + 0.5 * s), 0, m - 1).astype(np.intp)
           for s, n, m in zip(scales, new_dims, img.shape)]
    if img.flags.f_contiguous and not img.flags.c_contiguous:
        return _gather(img.T, *idx[::-1]).T
    return _gather(img, *idx)


def process_image(imggrid, dx, operations, ncells=None, num_threads=1):
    from imagemesh.image import np2pv

    numba.set_num_threads(min(num_threads, numba.config.NUMBA_NUM_THREADS))

    img = imggrid["data"]
    dims = imggrid.dimensions
    resolution = imggrid.spacing
    img = img.reshape(dims - np.array([1, 1, 1]), order="F")

    # get cells labels, and filter by n-largest (if requested)
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    if ncells:
        cell_labels = list(cell_labels[np.argsort(cell_counts)])
        if 0 in cell_labels:
            cell_labels.remove(0)
        cois = list(cell_labels[-ncells :])
        img = fastremap.mask_except(img, cois)
    else:
        cell_labels = list(cell_labels)
        if 0 in cell_labels: cell_labels.remove(0)

    # remap labels to smaller, sequential ints
    remapping = {int(c):i for i,c in enumerate([0] + cell_labels)}
    remap = lambda ids: [remapping[int(i)] for i in ids if i in remapping.keys()]
    img = fastremap.remap(img, remapping)
    img = fastremap.refit(img, value=len(cell_labels))

    # interpolate into the specified, isotropic grid with size dx
    img = resample_nearest(img, resolution, dx)
    print(f"image size: {img.shape}")
    resolution = np.array([dx]*3)
    roi = None

    # parse user specified operations, and iterate over them:
    operations = parse_operations(operations)
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
        if op=="upsample":
            dx /= kwargs["factor"]
            resolution = resolution / float(kwargs["factor"])
    
    print(f"processed! {img.shape}")

    # remap labels to smaller, sequential ints again, since many labels might have disappeared...
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    cell_labels = list(cell_labels[np.argsort(cell_counts)])
    if 0 in cell_labels: cell_labels.remove(0)

    remapping2 = {0:0}
    remapping2.update({int(c):i + 2 for i,c in enumerate(cell_labels)})
    img = fastremap.remap(img, remapping2, in_place=True)
    img = fastremap.refit(img, value=max(remapping2.values()))

    imggrid = np2pv(img, resolution)
    if roi is not None:
        imggrid["roimask"] = np.array(roi).flatten(order="F")


    combinedmap = {k:remapping2[v] for k,v in remapping.items() if v in remapping2.keys()}
    return imggrid, combinedmap, cell_labels, cell_counts