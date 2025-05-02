import numpy as np
import logging

from woodnet.datasets.pipelining.subselectors import PhysicalCenterCubeSubselector


class TestPhysicalCenterCubeSubselector:
    """Test class for PhysicalCenterCubeSubselector."""

    def test_smoke_physical_center_cube_subselector(self):
        """Simple smoke test to verify basic operation."""
        subselector = PhysicalCenterCubeSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0
        )
        data = np.random.rand(2, 20, 20, 20)
        result = subselector(data)
        assert result.shape == (2, 10, 10, 10)

    def test_cube_shape_computation(self):
        """Test the cube shape computation based on physical parameters."""
        # Test with clean division
        subselector = PhysicalCenterCubeSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0
        )
        assert subselector._cube_shape == (10, 10, 10)
        
        # Test with fractional division
        subselector = PhysicalCenterCubeSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=0.4
        )
        assert subselector._cube_shape == (25, 25, 25)

    def test_center_slices_computation(self):
        """Test computation of center slices."""
        # Test with even dimensions
        D, H, W = 20, 20, 20
        cubeshape = (10, 10, 10)
        slices = PhysicalCenterCubeSubselector._compute_center_slices(D, H, W, cubeshape)
        assert slices == (slice(5, 15), slice(5, 15), slice(5, 15))
        
        # Test with odd dimensions
        D, H, W = 21, 21, 21
        cubeshape = (11, 11, 11)
        slices = PhysicalCenterCubeSubselector._compute_center_slices(D, H, W, cubeshape)
        assert slices == (slice(5, 16), slice(5, 16), slice(5, 16))

    def test_subselector_with_array(self):
        """Test subselector functionality with a numpy array."""
        subselector = PhysicalCenterCubeSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0
        )
        
        # Create test data with batch dimension
        data = np.random.rand(2, 20, 20, 20)
        
        # Execute
        result = subselector(data)
        
        # Verify
        assert result.shape == (2, 10, 10, 10)
        # Check if center is extracted correctly
        np.testing.assert_array_equal(
            result, 
            data[:, 5:15, 5:15, 5:15]
        )

    def test_subselector_with_list(self):
        """Test subselector with a list of arrays."""
        subselector = PhysicalCenterCubeSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0
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
        assert result[0].shape == (2, 10, 10, 10)
        assert result[1].shape == (2, 10, 10, 10)
        np.testing.assert_array_equal(
            result[0],
            data_list[0][:, 5:15, 5:15, 5:15]
        )
        np.testing.assert_array_equal(
            result[1],
            data_list[1][:, 10:20, 10:20, 10:20]
        )

    def test_subselector_with_tuple(self):
        """Test subselector with a tuple of arrays."""
        subselector = PhysicalCenterCubeSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0
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
        assert result[0].shape == (2, 10, 10, 10)
        assert result[1].shape == (2, 10, 10, 10)
        np.testing.assert_array_equal(
            result[0],
            data_tuple[0][:, 5:15, 5:15, 5:15]
        )
        np.testing.assert_array_equal(
            result[1],
            data_tuple[1][:, 10:20, 10:20, 10:20]
        )

    def test_subselector_with_dict(self):
        """Test subselector with a dictionary of arrays."""
        subselector = PhysicalCenterCubeSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0
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
        assert result['image'].shape == (2, 10, 10, 10)
        assert result['mask'].shape == (2, 10, 10, 10)
        np.testing.assert_array_equal(
            result['image'],
            data_dict['image'][:, 5:15, 5:15, 5:15]
        )
        np.testing.assert_array_equal(
            result['mask'],
            data_dict['mask'][:, 10:20, 10:20, 10:20]
        )

    def test_action_logging(self, caplog):
        """Test that action logging works correctly."""
        caplog.set_level(logging.DEBUG)
        
        # Create subselector with logging enabled
        subselector = PhysicalCenterCubeSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0
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
        subselector = PhysicalCenterCubeSubselector(
            target_in_plane_length=10.0,
            input_voxel_size=1.0
        )
        str_rep = str(subselector)
        assert 'PhysicalCenterCubeSubselector' in str_rep
        assert 'target_in_plane_length=10.0' in str_rep
        assert 'input_voxel_size=1.0' in str_rep
