"""Tests for emimesh.process_image_data module."""
from imagemesh.handles import fill_handles
from imagemesh.pinches import remove_pinches

from emimesh.process_image_data import opdict, parse_operations, _parse_to_dict

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
