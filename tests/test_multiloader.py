import pytest
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from woodnet.trainer.multiloader import MultiLoader


class ConstantDataset(Dataset):
    """
    A simple dataset that returns constant tensor data with configurable shape and values.
    """
    def __init__(self, size=100, channels=3, height=32, width=32, values=None):
        """
        Initialize the dataset with specified parameters.
        
        Args:
            size: Number of samples in the dataset
            channels: Number of channels in each sample
            height: Height of each sample
            width: Width of each sample
            values: List of constant values to use (cycled through for each sample)
                   If None, uses [0, 1, 2] as default values
        """
        self.size = size
        self.shape = (channels, height, width)
        self.values = values if values is not None else [0, 1, 2]
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Use modulo to cycle through values for different samples
        value = self.values[idx % len(self.values)]
        # Create a tensor filled with the constant value
        data = torch.full(self.shape, value, dtype=torch.float32)
        # Return both data and the value as label for testing purposes
        return data, value


def test_constant_dataset():
    """Test the ConstantDataset implementation."""
    dataset = ConstantDataset(size=10, channels=2, height=16, width=16, values=[5, 10])
    
    # Check dataset length
    assert len(dataset) == 10
    
    # Check first sample
    data, label = dataset[0]
    assert data.shape == (2, 16, 16)
    assert data[0, 0, 0].item() == 5
    assert label == 5
    
    # Check second sample should have a different value
    data, label = dataset[1]
    assert data[0, 0, 0].item() == 10
    assert label == 10
    
    # Check that values cycle
    data, label = dataset[2]
    assert data[0, 0, 0].item() == 5
    assert label == 5


def test_multiloader_basic():
    """Test basic MultiLoader functionality with ConstantDatasets."""
    # Create two datasets with different values
    ds1 = ConstantDataset(size=10, values=[1], channels=3, height=32, width=32)
    ds2 = ConstantDataset(size=20, values=[2], channels=3, height=64, width=64)
    
    # Create loaders with different batch sizes
    loader1 = DataLoader(ds1, batch_size=2)
    loader2 = DataLoader(ds2, batch_size=4)
    
    # Create MultiLoader with equal weights
    multi_loader = MultiLoader(loaders=[loader1, loader2])
    
    # Check length
    assert len(multi_loader) == 5 + 5  # 10/2 + 20/4
    
    # Test getting batches
    for _ in range(10):
        batch_data, batch_labels = next(multi_loader)
        
        # Check that all values in a batch are the same
        unique_vals = torch.unique(batch_labels)
        assert len(unique_vals) == 1
        
        # Check that the value is either 1 or 2
        val = unique_vals.item()
        assert val in [1, 2]
        
        # Check batch shapes
        if val == 1:  # From loader1
            assert batch_data.shape == (2, 3, 32, 32)
        else:  # From loader2
            assert batch_data.shape == (4, 3, 64, 64)


def test_multiloader_with_weights():
    """Test MultiLoader with custom weights."""
    # Create two datasets
    ds1 = ConstantDataset(size=10, values=[1])
    ds2 = ConstantDataset(size=10, values=[2])
    
    # Create loaders
    loader1 = DataLoader(ds1, batch_size=1)
    loader2 = DataLoader(ds2, batch_size=1)
    
    # Create MultiLoader with custom weights (80% loader1, 20% loader2)
    multi_loader = MultiLoader(loaders=[loader1, loader2], weights=[0.8, 0.2], seed=42)
    
    # Sample 100 batches and count occurrences
    counts = {1: 0, 2: 0}
    for _ in range(100):
        _, batch_labels = next(multi_loader)
        val = batch_labels.item()
        counts[val] += 1
    
    # With enough samples, the distribution should be close to the weights
    # (allowing some variance due to randomness)
    assert counts[1] > counts[2]  # More samples from loader1
    assert counts[1] > 60  # Approximately 80% should be from loader1
    assert counts[2] < 40  # Approximately 20% should be from loader2


def test_multiloader_iterator_reset():
    """Test that iterators are properly reset when exhausted."""
    # Create small datasets
    ds1 = ConstantDataset(size=2, values=[1])
    ds2 = ConstantDataset(size=3, values=[2])
    
    # Create loaders with batch size 1
    loader1 = DataLoader(ds1, batch_size=1)
    loader2 = DataLoader(ds2, batch_size=1)
    
    # Create MultiLoader with fixed seed for deterministic testing
    multi_loader = MultiLoader(loaders=[loader1, loader2], seed=42)
    
    # Sample more batches than total dataset size to test iterator reset
    for _ in range(10):
        # This should not raise StopIteration
        batch_data, batch_labels = next(multi_loader)
        val = batch_labels.item()
        assert val in [1, 2]

