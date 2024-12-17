"""
Meta-test for synthetic dataset scaffolding.
"""
import random

from typing import Any

import torch

from woodnet.datasets.volumetric import TileDatasetBuilder, TileDataset
from woodnet.datasets.setup import InstanceFingerprint


def test_build_tile_dataset_from_synthetic_source(synthetic_dataset):
    data_configuration = synthetic_dataset.data_configuration
    instance_mapping = data_configuration.instance_mapping

    N_dataset = 4

    instances_ID = random.choices(
        list(instance_mapping.keys()), k=N_dataset
    )
    instance_mapping = {k : InstanceFingerprint(**v) for k, v in instance_mapping.items()}
    TileDatasetBuilder.instance_mapping = instance_mapping
    builder = TileDatasetBuilder()

    tileshape = (64, 64, 64)
    datasets = builder.build(instances_ID=instances_ID, phase='train',
                             tileshape=tileshape, transform_configurations=None)
    
    assert len(datasets) == N_dataset
    for item in datasets:
        assert isinstance(item, TileDataset)

    # test get item from dataset
    (data, label) = datasets[0][0]
    assert isinstance(data, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    # prepend dimension to emulate channel layout
    assert data.shape == (1, *tileshape)
    assert label.shape == (1,)
