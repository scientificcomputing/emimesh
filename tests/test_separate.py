"""separate_cells: contacts removed, pieces kept and connected, no new pinch, and blocks
with origin equal to the whole image."""

import numpy as np
import pytest
from imagemesh.pinches import _pinched_cubes
from nbmorph import separate_labels
from scipy import ndimage as ndi

from emimesh.process_image_data import opdict
from emimesh.separate import contact_mask, separate_cells


def pieces(img):
    """6-connected pieces per label."""
    return {int(v): ndi.label(img == v)[1] for v in np.unique(img) if v}


def check(before, after):
    assert not contact_mask(after).any()
    assert pieces(after) == pieces(before)
    assert not (_pinched_cubes(after) & ~_pinched_cubes(before)).any()


def crossing(n=13, h=6):
    img = np.zeros((n, n, h), np.int64)
    img[:, n // 2, 1] = 1
    img[n // 2, :, 2] = 2
    return img


@pytest.fixture(scope="module")
def cells():
    """Two crossings of one-voxel lines, and a short line on the face of a block."""
    a = np.zeros((40, 30, 12), np.int64)
    a[2:16, 8, 3] = 1
    a[8, 2:16, 4] = 2
    a[20:38, 18, 6] = 3
    a[29, 10:28, 7] = 4
    a[22:36, 2:9, 2:10] = 5
    a[24:32, 9, 5] = 6
    return a


def test_crossing_processes():
    """Neither contact voxel of two crossing lines can be emptied: one line is rerouted."""
    img = crossing()
    out = separate_cells(img, verbose=False)
    check(img, out)
    assert (out != img).sum() == 8


def test_cells(cells):
    out, stats = separate_cells(cells, verbose=False, return_stats=True)
    check(cells, out)
    assert stats[-1]["touching"] == 0


def test_block_equals_whole(cells):
    whole = separate_cells(cells, verbose=False)
    lo, hi, halo = np.array([20, 0, 0]), np.array([40, 30, 12]), 12
    wlo, whi = np.maximum(lo - halo, 0), np.minimum(hi + halo, cells.shape)
    block = separate_cells(
        cells[tuple(slice(a, b) for a, b in zip(wlo, whi))], origin=tuple(wlo), verbose=False
    )
    core = tuple(slice(a - w, b - w) for a, b, w in zip(lo, hi, wlo))
    np.testing.assert_array_equal(block[core], whole[tuple(slice(a, b) for a, b in zip(lo, hi))])


def test_workers_equal_serial(cells):
    np.testing.assert_array_equal(
        separate_cells(cells, workers=2, verbose=False), separate_cells(cells, verbose=False)
    )


def test_operations():
    assert opdict["separate_cells"] is separate_cells
    assert opdict["separate_labels"] is separate_labels
