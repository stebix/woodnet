import pytest
import numpy as np
import logging

from woodnet.datasets.pipelining.processors import ChannelSqueezingProcessor


class TestChannelSqueezingProcessor:


    def test_smoke_channel_squeezing_processor(self):
        """
        Simple smoke test to verify that ChannelSqueezingProcessor runs without errors.
        """
        # Test with single channel
        processor = ChannelSqueezingProcessor()
        data_single = np.random.rand(2, 1, 10, 10, 10)
        result_single = processor(data_single)
        assert result_single.shape == (2, 10, 10, 10)
        
        # Test with multiple channels
        processor_select = ChannelSqueezingProcessor(
            multichannel_strategy='select',
            channel_selection=0
        )
        data_multi = np.random.rand(2, 3, 10, 10, 10)
        result_multi = processor_select(data_multi)
        assert result_multi.shape == (2, 10, 10, 10)
        
        # Test with warn strategy
        processor_warn = ChannelSqueezingProcessor(multichannel_strategy='warn')
        result_warn = processor_warn(data_multi)
        assert result_warn.shape == (2, 10, 10, 10)        

    
    def test_single_channel(self):
        # Setup
        processor = ChannelSqueezingProcessor()
        data = np.random.rand(2, 1, 10, 20, 30)  # Shape: (batch, C, D, H, W)
        
        # Execute
        result = processor(data)
        
        # Verify
        assert result.shape == (2, 10, 20, 30)  # Shape should be: (batch, D, H, W)
    
    def test_multiple_channels_raise(self):
        # Setup
        processor = ChannelSqueezingProcessor(multichannel_strategy='raise')
        data = np.random.rand(2, 3, 10, 20, 30)  # Shape: (batch, C, D, H, W)
        
        # Execute and Verify
        with pytest.raises(ValueError) as excinfo:
            processor(data)
        assert 'Input data has 3 channels' in str(excinfo.value)
    
    def test_multiple_channels_select(self):
        # Setup
        processor = ChannelSqueezingProcessor(
            multichannel_strategy='select',
            channel_selection=1
        )
        data = np.random.rand(2, 3, 10, 20, 30)  # Shape: (batch, C, D, H, W)
        
        # Execute
        result = processor(data)
        
        # Verify
        assert result.shape == (2, 10, 20, 30)  # Shape should be: (batch, D, H, W)
        # Check if the correct channel was selected
        np.testing.assert_array_equal(result, data[:, 1, :, :, :])
    
    def test_multiple_channels_select_missing_channel_selection(self):
        # Setup
        processor = ChannelSqueezingProcessor(multichannel_strategy='select')
        data = np.random.rand(2, 3, 10, 20, 30)  # Shape: (batch, C, D, H, W)
        
        # Execute and Verify
        with pytest.raises(ValueError) as excinfo:
            processor(data)
        assert 'channel_selection must be provided' in str(excinfo.value)
    
    def test_multiple_channels_warn(self, caplog):
        # Setup
        caplog.set_level(logging.WARNING)
        processor = ChannelSqueezingProcessor(multichannel_strategy='warn')
        data = np.random.rand(2, 3, 10, 20, 30)  # Shape: (batch, C, D, H, W)
        
        # Execute
        result = processor(data)
        
        # Verify
        assert result.shape == (2, 10, 20, 30)  # Shape should be: (batch, D, H, W)
        # Check if the warning was logged
        assert 'Input data has 3 channels' in caplog.text
        # Check if channel 0 was selected
        np.testing.assert_array_equal(result, data[:, 0, :, :, :])
    
    def test_action_logging(self, caplog):
        # Setup
        caplog.set_level(logging.DEBUG)
        processor = ChannelSqueezingProcessor()
        data = np.random.rand(1, 1, 5, 5, 5)  # Shape: (batch, C, D, H, W)
        
        # Execute
        result = processor(data)
        
        # Verify
        assert 'channel processing action' in caplog.text
        assert str(data.shape) in caplog.text
        assert str(result.shape) in caplog.text
        
    # New tests for array sequence mapping functionality
    def test_with_list_of_arrays(self):
        # Setup
        processor = ChannelSqueezingProcessor()
        data_list = [
            np.random.rand(1, 1, 5, 5, 5),  # Single channel array
            np.random.rand(1, 1, 6, 6, 6)   # Another single channel array
        ]
        
        # Execute
        result = processor(data_list)
        
        # Verify
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (1, 5, 5, 5)
        assert result[1].shape == (1, 6, 6, 6)
        np.testing.assert_array_equal(result[0], data_list[0][:, 0, :, :, :])
        np.testing.assert_array_equal(result[1], data_list[1][:, 0, :, :, :])
    
    def test_with_tuple_of_arrays(self):
        # Setup
        processor = ChannelSqueezingProcessor()
        data_tuple = (
            np.random.rand(1, 1, 5, 5, 5),  # Single channel array
            np.random.rand(1, 1, 6, 6, 6)   # Another single channel array
        )
        
        # Execute
        result = processor(data_tuple)
        
        # Verify
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (1, 5, 5, 5)
        assert result[1].shape == (1, 6, 6, 6)
        np.testing.assert_array_equal(result[0], data_tuple[0][:, 0, :, :, :])
        np.testing.assert_array_equal(result[1], data_tuple[1][:, 0, :, :, :])
    
    def test_with_dict_of_arrays(self):
        # Setup
        processor = ChannelSqueezingProcessor()
        data_dict = {
            'image': np.random.rand(1, 1, 5, 5, 5),    # Single channel array
            'mask': np.random.rand(1, 1, 5, 5, 5)      # Another single channel array
        }
        
        # Execute
        result = processor(data_dict)
        
        # Verify
        assert isinstance(result, dict)
        assert len(result) == 2
        assert 'image' in result
        assert 'mask' in result
        assert result['image'].shape == (1, 5, 5, 5)
        assert result['mask'].shape == (1, 5, 5, 5)
        np.testing.assert_array_equal(result['image'], data_dict['image'][:, 0, :, :, :])
        np.testing.assert_array_equal(result['mask'], data_dict['mask'][:, 0, :, :, :])
    
    def test_mixed_channels_in_sequence(self):
        # Setup - test with a mix of single and multi-channel arrays
        processor = ChannelSqueezingProcessor(
            multichannel_strategy='select',
            channel_selection=1
        )
        data_list = [
            np.random.rand(1, 1, 5, 5, 5),  # Single channel array
            np.random.rand(1, 3, 6, 6, 6)   # Multi-channel array
        ]
        
        # Execute
        result = processor(data_list)
        
        # Verify
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (1, 5, 5, 5)
        assert result[1].shape == (1, 6, 6, 6)
        np.testing.assert_array_equal(result[0], data_list[0][:, 0, :, :, :])
        np.testing.assert_array_equal(result[1], data_list[1][:, 1, :, :, :])  # Channel 1 selected
    
    def test_array_sequence_raise_on_multichannel(self):
        # Setup
        processor = ChannelSqueezingProcessor(multichannel_strategy='raise')
        data_list = [
            np.random.rand(1, 1, 5, 5, 5),  # Single channel array - should process fine
            np.random.rand(1, 3, 6, 6, 6)   # Multi-channel array - should raise error
        ]
        
        # Execute and Verify
        with pytest.raises(ValueError) as excinfo:
            processor(data_list)
        assert 'Input data has 3 channels' in str(excinfo.value)
    
    def test_invalid_input_type(self):
        # Setup
        processor = ChannelSqueezingProcessor()
        invalid_data = "not an array or sequence"
        
        # Execute and Verify
        with pytest.raises(TypeError) as excinfo:
            processor(invalid_data)
        assert 'Expected a sequence of arrays or a single array' in str(excinfo.value)
