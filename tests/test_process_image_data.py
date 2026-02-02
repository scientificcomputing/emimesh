"""Tests for emimesh.process_image_data module."""
import numpy as np
from unittest.mock import patch
from emimesh.process_image_data import (
    mergecells, ncells, dilate, erode, smooth, removeislands, 
    opdict, parse_operations, _parse_to_dict
)

class TestImageOperations:
    """Test individual image processing operations."""
    
    def test_mergecells_basic(self):
        """Test basic cell merging."""
        img = np.array([[[1, 1, 2], [1, 3, 2], [4, 4, 4]]], dtype=np.uint32)
        labels = [1, 2]
        
        result = mergecells(img, labels)
        
        # All 1s and 2s should become the first label (1)
        non_zero_values = result[result > 0]
        unique_values = np.unique(non_zero_values)
        
        # Should only have values 1, 3, 4 (1 and 2 merged to 1)
        assert set(unique_values) == {1, 3, 4}
        assert 3 in result  # 3 should remain unchanged
        assert 4 in result  # 4 should remain unchanged
    
    def test_ncells_basic(self):
        """Test keeping only N largest cells."""
        img = np.array([[[1, 1, 2], [1, 3, 2], [4, 4, 4]]], dtype=np.uint32)
        
        result = ncells(img, ncells=2)
        
        # Should keep only background (0) and the two largest cells (1 and 4)
        assert np.allclose(np.unique(result), np.array([0, 1,4]))
    
    def test_ncells_with_keep_labels(self):
        """Test keeping specific cells regardless of size."""
        img = np.array([[[1, 1, 2], [1, 3, 2], [4, 4, 4]]], dtype=np.uint32)
        keep_labels = [2]
        
        result = ncells(img, ncells=1, keep_cell_labels=keep_labels)
        
        # Should only have background (0) and the kept label (2)
        assert np.allclose(np.unique(result), np.array([0, 2]))
    
    def test_removeislands_basic(self):
        """Test removing small islands."""
        # Create an image with small and large connected components
        img = np.zeros((10, 10, 10), dtype=np.uint32)
        img[2:4, 2:4, 2:4] = 1  # Small island (8 voxels)
        img[6:9, 6:9, 6:9] = 2  # Large island (27 voxels)
        
        result = removeislands(img, minsize=10)
        
        # Small island should be removed, large one should remain
        assert 1 not in np.unique(result)
        assert 2 in np.unique(result)
    

class TestOperationDictionary:
    """Test the operation dictionary."""
    
    def test_opdict_contains_all_operations(self):
        """Test that opdict contains all expected operations."""
        expected_ops = ["merge", "smooth", "dilate", "erode", "removeislands", "ncells"]
        
        for op in expected_ops:
            assert op in opdict
            assert callable(opdict[op])


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


class TestImageProcessingIntegration:
    """Integration tests for image processing operations."""
    
    def test_dilate_operation(self):
        """Test dilation operation."""
        img = np.zeros((10, 10, 10), dtype=np.uint32)
        img[4:6, 4:6, 4:6] = 1
        
        # Mock nbmorph.dilate_labels_spherical to avoid dependency
        with patch('emimesh.process_image_data.nbmorph') as mock_nbmorph:
            mock_nbmorph.dilate_labels_spherical.return_value = img  # Return same for simplicity
            
            result = dilate(img, radius=2)
            
            mock_nbmorph.dilate_labels_spherical.assert_called_once_with(img, radius=2)
    
    def test_erode_operation(self):
        """Test erosion operation."""
        img = np.ones((10, 10, 10), dtype=np.uint32)
        
        # Mock nbmorph.erode_labels_spherical to avoid dependency
        with patch('emimesh.process_image_data.nbmorph') as mock_nbmorph:
            mock_nbmorph.erode_labels_spherical.return_value = img  # Return same for simplicity
            
            result = erode(img, radius=2)
            
            mock_nbmorph.erode_labels_spherical.assert_called_once_with(img, radius=2)
    
    def test_smooth_operation(self):
        """Test smoothing operation."""
        img = np.ones((10, 10, 10), dtype=np.uint32)
        
        # Mock nbmorph.smooth_labels_spherical to avoid dependency
        with patch('emimesh.process_image_data.nbmorph') as mock_nbmorph:
            mock_nbmorph.smooth_labels_spherical.return_value = img  # Return same for simplicity
            
            result = smooth(img, iterations=5, radius=3)
            
            mock_nbmorph.smooth_labels_spherical.assert_called_once_with(
                img, radius=3, iterations=5, dilate_radius=3
            )