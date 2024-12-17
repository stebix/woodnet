"""
Meta-test for synthetic dataset scaffolding.
"""
import random

import torch

import pytest

from woodnet.datasets.volumetric import TileDatasetBuilder, TileDataset

from tests.scaffolding.syntheticdata import extract_classlabel_mapping, ClassSpecification


def test_extract_classlabel_mapping_with_inconsistent_specs():
    specs = [
        ClassSpecification(name='acer', label=0, groups={'red', 'green'}, instances_per_group=1),
        ClassSpecification(name='pinus', label=1, groups={'blue', 'green'}, instances_per_group=1),
        ClassSpecification(name='pinus', label=2, groups={'blue', 'green'}, instances_per_group=1),
    ]
    with pytest.raises(ValueError):
        extract_classlabel_mapping(specs)



def test_build_tile_dataset_from_synthetic_source(synthetic_dataset):
    instance_mapping = synthetic_dataset.instance_mapping
    N_dataset = 4
    instances_ID = random.choices(
        list(instance_mapping.keys()), k=N_dataset
    )
    # Appropriately prepare the builder class. It was initialized with uninformative
    # mock data, so we need to replace the instance mapping with the one from the
    # synthetic dataset. Also, the internal path is relevant.
    TileDatasetBuilder.instance_mapping = instance_mapping
    TileDatasetBuilder.internal_path = synthetic_dataset.internal_path
    TileDatasetBuilder.classlabel_mapping = synthetic_dataset.classlabel_mapping
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
