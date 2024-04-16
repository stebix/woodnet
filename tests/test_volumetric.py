import logging
import torch
import pytest

from woodnet.datasets import DEFAULT_CLASSLABEL_MAPPING
from woodnet.datasets.volumetric_inference import TransformedTileDataset, ParametrizedTransform


def normalize(inputs):
    return (inputs - 110) / 950


def zeroize(inputs):
    return 0 * inputs


def oneize(inputs):
    # satbility safeguard if value is very close to 0
    epsilon = 1e-5
    return (inputs + epsilon) / (inputs + epsilon)

@pytest.mark.slow
@pytest.mark.parametrize(
        'name,transform,const',
        [('Zeroize', zeroize, torch.tensor(0.0)),
         ('Oneize', oneize, torch.tensor(1.0))]
)
def test_correct_workings_of_parametrized_transform(caplog, name, transform, const):
    caplog.set_level(logging.DEBUG)
    ptf = ParametrizedTransform(name=name, parameters={}, transform=transform)
    conf = {
        'path' : '/home/jannik/storage/wood/custom/CT10.zarr',
        'tileshape' : (128, 128, 128),
        'transformer' : normalize,
        'classlabel_mapping' : DEFAULT_CLASSLABEL_MAPPING
    }
    dataset = TransformedTileDataset(**conf)
    dataset.parametrized_transform = ptf
    for data, _ in dataset:
        assert torch.allclose(data, const)