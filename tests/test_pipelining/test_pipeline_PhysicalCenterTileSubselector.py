import pytest
import numpy as np
import logging

from woodnet.datasets.pipelining.subselectors import PhysicalCenterTileSubselector, ZSpacingStrategy


class TestPhysicalCenterTileSubselector:
    """Test class for PhysicalCenterTileSubselector."""

    def test_smoke_physical_center_tile_subselector(self):
        """Simple smoke test to verify basic operation."""
        # Test with TIGHT strategy
        subselector_tight = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=5,
            z_spacing_strategy=ZSpacingStrategy.TIGHT
        )
        data = np.random.rand(2, 20, 20, 20)
        result = subselector_tight(data)
        assert result.shape == (2, 5, 10, 10)
        
        # Test with SPREAD strategy
        subselector_spread = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=5,
            z_spacing_strategy=ZSpacingStrategy.SPREAD
        )
        result = subselector_spread(data)
        assert result.shape == (2, 5, 10, 10)

    def test_in_plane_shape_computation(self):
        """Test in-plane shape computation based on physical parameters."""
        # Test with clean division
        subselector = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=5
        )
        assert subselector._in_plane_shape == (10, 10)
        
        # Test with fractional division
        subselector = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=0.4,
            target_slice_count=5
        )
        assert subselector._in_plane_shape == (25, 25)

    def test_center_slices_computation(self):
        """Test computation of center slices for in-plane dimensions."""
        # Test with even dimensions
        H, W = 20, 20
        tileshape = (10, 10)
        slices = PhysicalCenterTileSubselector._compute_center_slices(H, W, tileshape)
        assert slices == (slice(5, 15), slice(5, 15))
        
        # Test with odd dimensions
        H, W = 21, 21
        tileshape = (11, 11)
        slices = PhysicalCenterTileSubselector._compute_center_slices(H, W, tileshape)
        assert slices == (slice(5, 16), slice(5, 16))

    def test_z_indices_tight_strategy(self):
        """Test computation of z indices with TIGHT strategy."""
        D = 20
        target_slice_count = 5
        z_indices = PhysicalCenterTileSubselector._compute_z_indices(
            D, target_slice_count, ZSpacingStrategy.TIGHT
        )
        # Should be centered around the middle (D//2 = 10)
        expected_indices = np.array([8, 9, 10, 11, 12])
        np.testing.assert_array_equal(z_indices, expected_indices)

    def test_z_indices_spread_strategy(self):
        """Test computation of z indices with SPREAD strategy."""
        D = 20
        target_slice_count = 5
        z_indices = PhysicalCenterTileSubselector._compute_z_indices(
            D, target_slice_count, ZSpacingStrategy.SPREAD
        )
        # Should be evenly distributed
        expected_indices = np.array([0, 5, 10, 15, 19])
        np.testing.assert_array_equal(z_indices, expected_indices)

    def test_invalid_z_spacing_strategy(self):
        """Test error handling with invalid z spacing strategy."""
        with pytest.raises(ValueError) as excinfo:
            PhysicalCenterTileSubselector._compute_z_indices(
                20, 5, 'invalid_strategy'
            )
        assert 'Invalid z_spacing_strategy' in str(excinfo.value)

    def test_subselector_with_array(self):
        """Test subselector functionality with a numpy array."""
        subselector = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=5
        )
        
        # Create test data with channel dimension
        data = np.random.rand(2, 20, 20, 20)
        
        # Execute
        result = subselector(data)
        
        # Verify
        assert result.shape == (2, 5, 10, 10)
        # Check if center is extracted correctly
        expected_z_indices = np.array([8, 9, 10, 11, 12])
        np.testing.assert_array_equal(
            result, 
            data[:, expected_z_indices, 5:15, 5:15]
        )

    def test_subselector_with_list(self):
        """Test subselector with a list of arrays."""
        subselector = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=5
        )
        
        # Create test data
        data_list = [
            np.random.rand(2, 20, 20, 20),
            np.random.rand(2, 30, 30, 30)
        ]
        
        # Execute
        result = subselector(data_list)
        
        # Verify
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (2, 5, 10, 10)
        assert result[1].shape == (2, 5, 10, 10)
        # z indices centered around the middle D // 2 for D in {20, 30}
        expected_z_indices_0 = np.array([8, 9, 10, 11, 12])
        expected_z_indices_1 = np.array([13, 14, 15, 16, 17])
        
        np.testing.assert_array_equal(
            result[0],
            data_list[0][:, expected_z_indices_0, 5:15, 5:15]
        )
        np.testing.assert_array_equal(
            result[1],
            data_list[1][:, expected_z_indices_1, 10:20, 10:20]
        )

    def test_subselector_with_tuple(self):
        """Test subselector with a tuple of arrays."""
        subselector = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=5
        )
        
        # Create test data
        data_tuple = (
            np.random.rand(2, 20, 20, 20),
            np.random.rand(2, 30, 30, 30)
        )
        
        # Execute
        result = subselector(data_tuple)
        
        # Verify
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (2, 5, 10, 10)
        assert result[1].shape == (2, 5, 10, 10)

    def test_subselector_with_dict(self):
        """Test subselector with a dictionary of arrays."""
        subselector = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=5
        )
        
        # Create test data
        data_dict = {
            'image': np.random.rand(2, 20, 20, 20),
            'mask': np.random.rand(2, 30, 30, 30)
        }
        
        # Execute
        result = subselector(data_dict)
        
        # Verify
        assert isinstance(result, dict)
        assert len(result) == 2
        assert 'image' in result
        assert 'mask' in result
        assert result['image'].shape == (2, 5, 10, 10)
        assert result['mask'].shape == (2, 5, 10, 10)

    def test_z_spacing_strategy_from_string(self):
        """Test initialization with string z_spacing_strategy."""
        subselector = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=5,
            z_spacing_strategy='spread'
        )
        assert subselector.z_spacing_strategy == ZSpacingStrategy.SPREAD

    def test_too_many_slices_error(self):
        """Test error handling when requesting too many slices."""
        subselector = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=25
        )
        data = np.random.rand(2, 20, 20, 20)
        
        with pytest.raises(ValueError) as excinfo:
            subselector(data)
        assert 'requested target slice count' in str(excinfo.value)

    def test_action_logging(self, caplog):
        """Test that action logging works correctly."""
        caplog.set_level(logging.DEBUG)
        
        # Create subselector with logging enabled
        subselector = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=5
        )
        subselector.log_action = True
        
        # Create data and execute
        data = np.random.rand(2, 20, 20, 20)
        result = subselector(data)
        
        # Verify logging
        assert any('action' in record.message for record in caplog.records)
        assert any(str(data.shape) in record.message for record in caplog.records)
        assert any(str(result.shape) in record.message for record in caplog.records)

    def test_str_representation(self):
        """Test string representation of the subselector."""
        subselector = PhysicalCenterTileSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0,
            target_slice_count=5
        )
        str_rep = str(subselector)
        assert 'PhysicalCenterTileSubselector' in str_rep
        assert 'target_in_plane_length=10.0' in str_rep
        assert 'input_voxel_size=1.0' in str_rep
        assert 'target_slice_count=5' in str_rep
        assert 'z_spacing_strategy' in str_rep
