import pyvista as pv
import numpy as np
import fastremap


def np2pv(arr, resolution, roimask=None):
    grid = pv.ImageData(
        dimensions=arr.shape + np.array((1, 1, 1)), spacing=resolution, origin=(0, 0, 0)
    )
    grid[f"data"] = arr.flatten(order="F")
    if roimask is not None:
        grid["roimask"] = roimask.flatten(order="F")
    return grid


def get_cell_frequencies(img):
    # dask is slow, see: https://github.com/dask/dask/issues/10510
    # cell_labels, cell_counts = dask.compute(da.unique(img, return_counts=True))[0]
    cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
    indices = np.argsort(cell_counts)
    return np.vstack([cell_labels[indices], cell_counts[indices]])


def get_bounding_box(mesh, eps=0):

    eps = np.vstack([np.ones(3), - np.ones(3)]).T * eps
    bounds = np.array(mesh.bounds).reshape(3,2)
    return pv.Box((bounds + eps).flatten())
