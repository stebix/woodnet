import numpy as np
import logging

from unittest.mock import patch

import pytest

from woodnet.datasets.pipelining.subselectors import CentroidCubeSubselector


class TestCentroidCubeSubselector:
    """Test class for CentroidCubeSubselector."""

    def test_smoke_centroid_cube_subselector(self):
        """Simple smoke test to verify basic operation."""
        # Test with no cube limit
        subselector = CentroidCubeSubselector()
        # Create data where cube size will be smaller than depth (to generate multiple cubes)
        data = np.random.rand(2, 30, 20, 20)
        
        # Mock compute_centroid_square to return a fixed square size
        with patch('woodnet.datasets.pipelining.subselectors.compute_centroid_square', 
                   return_value=(5, 5, 10, 10)):
            with patch('woodnet.datasets.pipelining.subselectors.to_slices', 
                      return_value=(slice(5, 15), slice(5, 15))):
                result = subselector(data)
                
                # Verify multiple cubes are returned as a list
                assert isinstance(result, list)
                assert len(result) == 3  # 30 depth / 10 cube size = 3 cubes
                assert result[0].shape == (2, 10, 10, 10)
                assert result[1].shape == (2, 10, 10, 10)
                assert result[2].shape == (2, 10, 10, 10)

    def test_cube_limit_functionality(self):
        """Test the cube_limit parameter functionality."""
        # Create data where multiple cubes could be extracted
        data = np.random.rand(2, 40, 20, 20)
        
        # Test with cube_limit=2 (should return only 2 cubes even though 4 would fit)
        subselector_limited = CentroidCubeSubselector(cube_limit=2)
        
        with patch('woodnet.datasets.pipelining.subselectors.compute_centroid_square', 
                   return_value=(5, 5, 10, 10)):
            with patch('woodnet.datasets.pipelining.subselectors.to_slices', 
                      return_value=(slice(5, 15), slice(5, 15))):
                result_limited = subselector_limited(data)
                
                # Verify only 2 cubes are returned
                assert isinstance(result_limited, list)
                assert len(result_limited) == 2
                assert result_limited[0].shape == (2, 10, 10, 10)
                assert result_limited[1].shape == (2, 10, 10, 10)
                
                # Test with no limit (should return 4 cubes)
                subselector_unlimited = CentroidCubeSubselector()
                result_unlimited = subselector_unlimited(data)
                
                # Verify 4 cubes are returned
                assert isinstance(result_unlimited, list)
                assert len(result_unlimited) == 4
                assert result_unlimited[0].shape == (2, 10, 10, 10)

    def test_output_squeezing(self):
        """Test that output is squeezed when only one cube is selected."""
        # Create data where cube size matches depth (one cube will fit)
        data = np.random.rand(2, 10, 20, 20)
        
        # Test with no limit
        subselector = CentroidCubeSubselector()
        
        with patch('woodnet.datasets.pipelining.subselectors.compute_centroid_square', 
                   return_value=(5, 5, 10, 10)):
            with patch('woodnet.datasets.pipelining.subselectors.to_slices', 
                      return_value=(slice(5, 15), slice(5, 15))):
                result = subselector(data)
                
                # Verify result is a squeezed array, not a list
                assert not isinstance(result, list)
                assert result.shape == (2, 10, 10, 10)
                
                # Test with cube_limit=1 (explicitly limiting to one cube)
                subselector_limited = CentroidCubeSubselector(cube_limit=1)
                result_limited = subselector_limited(data)
                
                # Verify result is a squeezed array, not a list
                assert not isinstance(result_limited, list)
                assert result_limited.shape == (2, 10, 10, 10)

    def test_subselector_with_array_multiple_cubes(self):
        """Test that multiple cubes are correctly extracted with proper spacing."""
        # Create test data where cube size will be smaller than depth
        data = np.zeros((2, 40, 20, 20))
        # Fill with increasing values to verify correct extraction
        for i in range(40):
            for j in range(20):
                for k in range(20):
                    data[:, i, j, k] = i * 100 + j * 10 + k
        
        # Mock to return a cube size of 10x10
        with patch('woodnet.datasets.pipelining.subselectors.compute_centroid_square', 
                   return_value=(5, 5, 10, 10)):
            with patch('woodnet.datasets.pipelining.subselectors.to_slices', 
                      return_value=(slice(5, 15), slice(5, 15))):
                # Execute with no cube limit
                subselector = CentroidCubeSubselector()
                result = subselector(data)
                
                # Verify multiple cubes are returned
                assert isinstance(result, list)
                assert len(result) == 4  # 40 depth / 10 cube size = 4 cubes
                
                # Verify the spacing between cubes is correct
                # First cube should start at z=0
                for z in range(10):
                    assert result[0][0, z, 0, 0] == z * 100 + 5 * 10 + 5
                
                # Second cube should start at z=10
                for z in range(10):
                    assert result[1][0, z, 0, 0] == (z + 10) * 100 + 5 * 10 + 5
                
                # Third cube should start at z=20
                for z in range(10):
                    assert result[2][0, z, 0, 0] == (z + 20) * 100 + 5 * 10 + 5

    def test_subselector_with_list(self):
        """Test subselector with a list of arrays."""
        subselector = CentroidCubeSubselector()
        
        # Create test data where multiple cubes will be extracted
        data_list = [
            np.random.rand(2, 20, 20, 20),
            np.random.rand(2, 30, 30, 30)
        ]
        
        # Execute with mocked centroid calculation
        with patch('woodnet.datasets.pipelining.subselectors.compute_centroid_square', 
                   side_effect=[(5, 5, 10, 10), (10, 10, 15, 15)]):
            with patch('woodnet.datasets.pipelining.subselectors.to_slices', 
                      side_effect=[(slice(5, 15), slice(5, 15)), (slice(10, 25), slice(10, 25))]):
                result = subselector(data_list)
                
                # Verify
                assert isinstance(result, list)
                assert len(result) == 2
                # For first array, expect 2 cubes (20/10=2)
                assert isinstance(result[0], list)
                assert len(result[0]) == 2
                assert result[0][0].shape == (2, 10, 10, 10)
                # For second array, expect 2 cubes (30/15=2)
                assert isinstance(result[1], list)
                assert len(result[1]) == 2
                assert result[1][0].shape == (2, 15, 15, 15)

    def test_subselector_with_tuple(self):
        """Test subselector with a tuple of arrays."""
        subselector = CentroidCubeSubselector()
        
        # Create test data
        data_tuple = (
            np.random.rand(2, 20, 20, 20),
            np.random.rand(2, 30, 30, 30)
        )
        
        # Execute with mocked centroid calculation
        with patch('woodnet.datasets.pipelining.subselectors.compute_centroid_square', 
                   side_effect=[(5, 5, 10, 10), (10, 10, 15, 15)]):
            with patch('woodnet.datasets.pipelining.subselectors.to_slices', 
                      side_effect=[(slice(5, 15), slice(5, 15)), (slice(10, 25), slice(10, 25))]):
                result = subselector(data_tuple)
                
                # Verify
                assert isinstance(result, tuple)
                assert len(result) == 2
                # For first array, expect 2 cubes (20/10=2)
                assert isinstance(result[0], list)
                assert len(result[0]) == 2
                assert result[0][0].shape == (2, 10, 10, 10)
                # For second array, expect 2 cubes (30/15=2)
                assert isinstance(result[1], list)
                assert len(result[1]) == 2
                assert result[1][0].shape == (2, 15, 15, 15)

    def test_subselector_with_dict(self):
        """Test subselector with a dictionary of arrays."""
        subselector = CentroidCubeSubselector()
        
        # Create test data
        data_dict = {
            'image': np.random.rand(2, 20, 20, 20),
            'mask': np.random.rand(2, 30, 30, 30)
        }
        
        # Execute with mocked centroid calculation
        with patch('woodnet.datasets.pipelining.subselectors.compute_centroid_square', 
                   side_effect=[(5, 5, 10, 10), (10, 10, 15, 15)]):
            with patch('woodnet.datasets.pipelining.subselectors.to_slices', 
                      side_effect=[(slice(5, 15), slice(5, 15)), (slice(10, 25), slice(10, 25))]):
                result = subselector(data_dict)
                
                # Verify
                assert isinstance(result, dict)
                assert len(result) == 2
                assert 'image' in result
                assert 'mask' in result
                # For image array, expect 2 cubes (20/10=2)
                assert isinstance(result['image'], list)
                assert len(result['image']) == 2
                assert result['image'][0].shape == (2, 10, 10, 10)
                # For mask array, expect 2 cubes (30/15=2)
                assert isinstance(result['mask'], list)
                assert len(result['mask']) == 2
                assert result['mask'][0].shape == (2, 15, 15, 15)

    def test_action_logging(self, caplog):
        """Test that action logging works correctly with both single and multiple outputs."""
        caplog.set_level(logging.DEBUG)
        
        # Create subselector with logging enabled
        subselector = CentroidCubeSubselector()
        subselector.log_action = True
        
        # Test with single output (squeezed)
        data_single = np.random.rand(2, 10, 20, 20)
        with patch('woodnet.datasets.pipelining.subselectors.compute_centroid_square', 
                   return_value=(5, 5, 10, 10)):
            with patch('woodnet.datasets.pipelining.subselectors.to_slices', 
                      return_value=(slice(5, 15), slice(5, 15))):
                caplog.clear()
                result_single = subselector(data_single)
                
                # Verify logging for single output
                assert any('action' in record.message for record in caplog.records)
                assert any(str(data_single.shape) in record.message for record in caplog.records)
                assert any(str(result_single.shape) in record.message for record in caplog.records)
        
        # Test with multiple outputs
        data_multi = np.random.rand(2, 30, 20, 20)
        with patch('woodnet.datasets.pipelining.subselectors.compute_centroid_square', 
                   return_value=(5, 5, 10, 10)):
            with patch('woodnet.datasets.pipelining.subselectors.to_slices', 
                      return_value=(slice(5, 15), slice(5, 15))):
                caplog.clear()
                result_multi = subselector(data_multi)
                
                # Verify logging for multiple outputs
                assert any('action' in record.message for record in caplog.records)
                assert any(str(data_multi.shape) in record.message for record in caplog.records)
                assert any(f'N={len(result_multi)}' in record.message for record in caplog.records)

    def test_str_representation(self):
        """Test string representation of the subselector."""
        subselector = CentroidCubeSubselector(cube_limit=2)
        str_rep = str(subselector)
        assert 'CentroidCubeSubselector' in str_rep
        assert 'cube_limit=2' in str_rep
        
        subselector = CentroidCubeSubselector()
        str_rep = str(subselector)
        assert 'CentroidCubeSubselector' in str_rep
        assert 'cube_limit=None' in str_rep

    def test_compute_z_slices(self):
        """Test the _compute_z_slices method explicitly."""
        # Test normal case (D > size)
        D, size = 30, 10
        zslices = CentroidCubeSubselector._compute_z_slices(D, size, None)
        assert len(zslices) == 3
        assert zslices[0] == slice(0, 10)
        assert zslices[1] == slice(10, 20)
        assert zslices[2] == slice(20, 30)
        
        # Test with cube limit
        zslices_limited = CentroidCubeSubselector._compute_z_slices(D, size, 2)
        assert len(zslices_limited) == 2
        
        # Test error case (D < size)
        D, size = 5, 10
        with pytest.raises(ValueError) as exc_info:
            CentroidCubeSubselector._compute_z_slices(D, size, None)
            assert str(exc_info.value) == f'cannot fit cube of size {size} into data of depth {D}'