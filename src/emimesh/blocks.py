"""Helpers for meshing a label image in blocks (see split_blocks.py)."""

import numpy as np


def merge_surfaces(surfs, eps, planes):
    """
    Merge the points of the block surfaces that lie on the cut planes z in
    planes (coincident within eps). Returns the merged points and, per block,
    its quads in merged point ids. Other coincident points are not merged:
    surface nets splits pinches into coincident points, and merging them
    gives non-manifold edges whose points are locked in the smoothing.
    """
    pts = np.vstack([s.points for s in surfs]).astype(np.float64)
    key = np.round(pts / eps)
    off = ~np.isclose(pts[:, 2][:, None], planes, rtol=0, atol=eps).any(axis=1)
    key[off] = np.column_stack([np.zeros((off.sum(), 2)), -1 - np.arange(off.sum())])
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    offsets = np.cumsum([0] + [s.n_points for s in surfs])
    quads = [inv[s.regular_faces + o] for s, o in zip(surfs, offsets[:-1])]
    return pts[idx], quads
