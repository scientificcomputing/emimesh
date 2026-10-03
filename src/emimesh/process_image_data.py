import numpy as np
import yaml
import fastremap
import dask
import cc3d
import nbmorph
from pathlib import Path

from emimesh.handles import fill_handles
from emimesh.pinches import remove_pinches

dask.config.set({"array.chunk-size": "1024 MiB"})

def mergecells(img, labels):
    print(f"merging cells: {labels},  ({img.shape})")
    img = np.where(np.isin(img, labels), labels[0], img)
    return img

def ncells(img, ncells, keep_cell_labels=None):
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    cell_labels = cell_labels[np.argsort(cell_counts)][::-1]
    if keep_cell_labels is None: 
        cois = set()
    else:
        cois = set(keep_cell_labels)
    for cid in cell_labels:
        if len(cois) >= ncells: break
        cois.add(cid)
    img = np.where(np.isin(img, list(cois)), img, 0)
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

def erode(img, radius, labels=None, struct_sequence="DDB"):
    print(f"eroding cells,  ({img.shape})")
    if labels is None:
        img = nbmorph.erode_labels_spherical(img, radius=radius, struct_sequence=struct_sequence)
    else:
        vipimg = np.where(np.isin(img, labels), img, 0)
        vipimg = erode(vipimg, radius=radius)
        orig_wo_vips = np.where(np.isin(img, labels), 0, img)
        img = np.where(orig_wo_vips > vipimg, orig_wo_vips, vipimg)
    return img

def mode(img, iterations=1):
    print(f"mode,  ({img.shape})")
    for i in range(iterations): 
        img = nbmorph.mode_box(img)
    return img

def zero_edges(img):
    print(f"zero_edges,  ({img.shape})")
    img = nbmorph.zero_label_edges_box(img)
    return img

def upsample(img, factor):
    from scipy.ndimage import zoom
    return zoom(img, factor, order=0)

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
    

def process_image(imggrid, dx, operations, ncells=None, num_threads=1):
    import dask.array as da
    from dask_image.ndinterp import affine_transform
    import numba
    from functools import partial
    from emimesh.utils import np2pv

    numba.set_num_threads(num_threads)

    img = imggrid["data"]
    dims = imggrid.dimensions
    resolution = imggrid.spacing
    img = img.reshape(dims - np.array([1, 1, 1]), order="F")
    img = da.from_array(img)

    # get cells labels, and filter by n-largest (if requested)
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    if ncells:
        cell_labels = list(cell_labels[np.argsort(cell_counts)])
        cell_labels.remove(0)
        cois = list(cell_labels[-ncells :])
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
    scale = np.diag([dx / r for r in resolution] + [1])
    new_dims = [int(d * r / dx) for d,r in zip(dims, resolution)]
    img = affine_transform(img, scale, output_shape=new_dims, order=0,
                           output_chunks=500)
    img = dask.compute(img, num_workers=num_threads)[0]
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
    cell_labels.remove(0)

    img = da.array(img)
    remapping2 = {0:0}
    remapping2.update({int(c):i + 2 for i,c in enumerate(cell_labels)})
    img = img.map_blocks(partial(fastremap.remap, table=remapping2), dtype=img.dtype)
    img = img.map_blocks(partial(fastremap.refit, value=max(remapping2.values())))
    img = dask.compute(img, num_workers=num_threads)[0]

    imggrid = np2pv(img, resolution)
    if roi is not None:
        imggrid["roimask"] = np.array(roi).flatten(order="F")


    combinedmap = {k:remapping2[v] for k,v in remapping.items() if v in remapping2.keys()}
    return imggrid, combinedmap, cell_labels, cell_counts