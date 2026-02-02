"""Global pytest configuration and fixtures for EMIMesh testing."""
import pytest
import numpy as np
import pyvista as pv
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_data():
    """Create sample image data for testing."""
    # Create a 3D array with some labeled regions
    data = np.zeros((50, 50, 50), dtype=np.uint32)
    
    # Add some labeled cells
    data[10:20, 10:20, 10:20] = 1
    data[30:40, 30:40, 30:40] = 2
    data[15:25, 35:45, 15:25] = 3
    
    return data


@pytest.fixture
def sample_resolution():
    """Sample resolution for testing."""
    return [18.0, 18.0, 18.0]


@pytest.fixture
def sample_pyvista_grid(sample_image_data, sample_resolution):
    """Create a sample PyVista grid for testing."""
    grid = pv.ImageData(
        dimensions=sample_image_data.shape + np.array([1, 1, 1]), 
        spacing=sample_resolution, 
        origin=(0, 0, 0)
    )
    grid["data"] = sample_image_data.flatten(order="F")
    return grid


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"
