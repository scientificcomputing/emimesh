import numpy as np
import yaml
import fastremap
import dask
import cc3d
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
    