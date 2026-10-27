"""Tests for emimesh.process_image_data module."""
import numpy as np
import pytest
from imagemesh.handles import fill_handles
from imagemesh.pinches import remove_pinches
from scipy.ndimage import affine_transform

from emimesh.process_image_data import opdict, parse_operations, _parse_to_dict, resample_nearest

class TestOperationDictionary:
    """Test the operation dictionary."""
    
    def test_opdict_contains_all_operations(self):
        """Test that opdict contains all expected operations."""
        expected_ops = ["merge", "smooth", "dilate", "erode", "removeislands", "ncells",
                        "remove_pinches"]
        
        for op in expected_ops:
            assert op in opdict
            assert callable(opdict[op])

    def test_topology_repair_operations(self):
        assert opdict["remove_pinches"] is remove_pinches
        assert opdict["fill_handles"] is fill_handles


class TestParseOperations:
    """Test operation parsing functionality."""
    
    def test_parse_to_dict_basic(self):
        """Test basic dictionary parsing."""
        values = ["key1='value1'", "key2=42", "key3=True"]
        
        result = _parse_to_dict(values)
        
        assert result["key1"] == "value1"
        assert result["key2"] == 42
        assert result["key3"] is True
    
    def test_parse_to_dict_with_lists(self):
        """Test parsing with list values."""
        values = ["labels='[1, 2, 3]'", "radius=5"]
        
        result = _parse_to_dict(values)
        
        assert result["labels"] == [1, 2, 3]
        assert result["radius"] == 5
    
    def test_parse_operations_basic(self):
        """Test basic operation parsing."""
        ops = [["merge", "labels='[1, 2]'", "radius=5"]]
        
        result = parse_operations(ops)
        
        assert len(result) == 1
        assert result[0][0] == "merge"
        assert result[0][1]["labels"] == [1, 2]
        assert result[0][1]["radius"] == 5
    
    def test_parse_operations_multiple(self):
        """Test parsing multiple operations."""
        ops = [
            ["merge", "labels='[1, 2]'"],
            ["removeislands", "minsize=100"],
            ["dilate", "radius=3"]
        ]
        
        result = parse_operations(ops)
        
        assert len(result) == 3
        assert result[0][0] == "merge"
        assert result[1][0] == "removeislands"
        assert result[2][0] == "dilate"


class TestResampleNearest:
    """Test nearest-neighbour resampling onto an isotropic grid."""

    @pytest.mark.parametrize("resolution, dx", [
        ((8, 8, 40), 16),
        ((8, 8, 40), 20),
        ((4, 4, 40), 12.5),
        ((16, 16, 16), 8),
        ((7.3, 9.1, 33.3), 6),
    ])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_matches_scipy_affine_transform(self, resolution, dx, order):
        """Test that the result matches scipy's affine_transform (order=0, voxel centers)."""
        rng = np.random.default_rng(0)
        img = np.asarray(rng.integers(0, 50, (37, 41, 29), dtype=np.uint32), order=order)

        scale = np.diag([dx / r for r in resolution] + [1])
        scale[:3, 3] = 0.5 * np.diag(scale)[:3] - 0.5
        new_dims = [int(n * r / dx) for n, r in zip(img.shape, resolution)]
        expected = affine_transform(img, scale, output_shape=new_dims, order=0, mode="nearest")

        result = resample_nearest(img, resolution, dx)

        assert result.dtype == img.dtype
        np.testing.assert_array_equal(result, expected)

    def test_identity(self):
        """Test that resampling to the original isotropic spacing is a no-op."""
        img = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)

        result = resample_nearest(img, (2, 2, 2), 2)

        np.testing.assert_array_equal(result, img)
