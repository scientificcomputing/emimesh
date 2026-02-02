"""Tests for emimesh.extract_surfaces module."""
import numpy as np
import pyvista as pv

from emimesh.extract_surfaces import (
    extract_surface, create_balanced_csg_tree,
    clean_mesh_nan_points
)
from emimesh.utils import np2pv


class TestExtractSurface:
    """Test surface extraction functionality."""
    
    def test_extract_surface_basic(self):
        """Test basic surface extraction."""
        # Create a simple test volume
        mask = np.zeros((10, 10, 10), dtype=np.uint32)
        mask[2:7, 2:8, 2:8] = 1  # Cube in the center
        
        grid = pv.ImageData(dimensions=mask.shape, spacing=(1,1,1), origin=(0, 0, 0))
        print(grid)
        print(mask)
        result = extract_surface(mask, grid, mesh_reduction_factor=2, taubin_smooth_iter=5)
        
        # Should return a valid mesh
        assert isinstance(result, pv.PolyData)
        assert result.is_manifold
        assert not np.isnan(result.points).any()
    
    
    def test_extract_surface_too_small(self):
        """Test surface extraction with too small volume."""
        mask = np.zeros((10, 10, 10), dtype=np.uint32)
        mask[5, 5, 5] = 1  # Single voxel
        
        grid = pv.ImageData(dimensions=(10, 10, 10), spacing=(1, 1, 1))
        
        result = extract_surface(mask, grid, mesh_reduction_factor=10, taubin_smooth_iter=5)
        
        assert result is False

class TestCSGTree:
    """Test CSG tree creation."""
    
    def test_create_balanced_csg_tree_single(self):
        """Test CSG tree creation with single surface."""
        surface_files = ["surface1.ply"]
        
        result = create_balanced_csg_tree(surface_files)
        
        assert result == "surface1.ply"
    
    def test_create_balanced_csg_tree_two(self):
        """Test CSG tree creation with two surfaces."""
        surface_files = ["surface1.ply", "surface2.ply"]
        
        result = create_balanced_csg_tree(surface_files)
        
        expected = {
            "operation": "union",
            "left": "surface1.ply",
            "right": "surface2.ply"
        }
        assert result == expected
    
    def test_create_balanced_csg_tree_multiple(self):
        """Test CSG tree creation with multiple surfaces."""
        surface_files = ["s1.ply", "s2.ply", "s3.ply", "s4.ply"]
        
        result = create_balanced_csg_tree(surface_files)
        
        # Should be a balanced tree structure
        assert result["operation"] == "union"
        assert "left" in result
        assert "right" in result
        
        # Left and right should each contain 2 surfaces
        assert isinstance(result["left"], dict) or isinstance(result["left"], str)
        assert isinstance(result["right"], dict) or isinstance(result["right"], str)