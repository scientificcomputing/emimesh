import pyvista as pv
import numpy as np
import dask
import dask.array as da


def np2pv(arr, resolution):
    grid = pv.ImageData(
        dimensions=arr.shape + np.array((1, 1, 1)), spacing=resolution, origin=(0, 0, 0)
    )
    grid[f"data"] = arr.flatten(order="F")
    return grid


def get_cell_frequencies(img):
    # dask is slow, see: https://github.com/dask/dask/issues/10510
    # cell_labels, cell_counts = dask.compute(da.unique(img, return_counts=True))[0]
    cell_labels, cell_counts = np.unique(img, return_counts=True)
    indices = np.argsort(cell_counts)
    return cell_labels[indices], cell_counts[indices]


def get_bounding_box(meshes, eps=0):
    max_p = -np.ones(3) * np.inf
    min_p = +np.ones(3) * np.inf

    for m in meshes:
        try:
            p = m.points
        except:
            p = m
        mmax = p.max(axis=0)
        max_p = np.array([max_p, mmax]).max(axis=0)
        mmin = p.min(axis=0)
        min_p = np.array([min_p, mmin]).min(axis=0)

    bounds = np.array([min_p + eps, max_p - eps]).T.flatten()
    return pv.Box(bounds)
