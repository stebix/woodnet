import random

import torch
import pytest

from woodnet.datasets.planar import EagerSliceDatasetBuilder



@pytest.mark.slow
def test_smoke_instantiation():
    builder = EagerSliceDatasetBuilder()

    datasets = builder.build(instances_ID=['CT10', 'CT12'],
                             phase='val', axis=0,
                             transform_configurations=None)
    dataset = torch.utils.data.ConcatDataset(datasets)
    element = random.choice(dataset)
    data, label = element
    assert data.ndim == 3, f'expecting 3D (C x H x W) data, but got ndim = {data.ndim}'
    assert label.ndim == 1, f'expecting 1D classification label, but got ndim = {label.ndim}'

