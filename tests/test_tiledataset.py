"""
test the full 3D tile dataset.

Usage hints:
Core scaffolding and setup of artificial data is done in the conftest.py file.
There, we create a module-scoped temporary directory that is used to store an realistic
raw dataset structure (i.e. numerical data and a data fingerprint) in the form of a zarr array.

On the builder class, we can patch the instance mapping and internal path to point to the
temporary directory and the data configuration file. This way, we can test the builder
in a controlled environment without the need to access the real data.
"""
import logging
import torch

import pytest

from woodnet.datasets.volumetric import TileDatasetBuilder, TileDataset
from woodnet.transformations import from_configurations
from woodnet.datasets.reader import ZarrReader


def zeroize(inputs):
    return 0 * inputs


def oneize(inputs):
    # satbility safeguard if value is very close to 0
    epsilon = 1e-5
    return (inputs + epsilon) / (inputs + epsilon)



def test_smoke_builder(mock_instance_mapping, internal_path):
    """General smoke test for the builder the produced datasets."""
    TileDatasetBuilder.instance_mapping = mock_instance_mapping
    TileDatasetBuilder.internal_path = internal_path
    TileDatasetBuilder.classlabel_mapping = {
        'acer' : 0,
        'pinus' : 1
    }

    phase = 'train'
    tileshape = (64, 64, 64)
    configs = [
        {'name' : 'Normalize', 'mean' : 0.0, 'std' : 1.0}
    ]
    IDs = ['picard', 'kirk']

    builder = TileDatasetBuilder()

    datasets = builder.build(IDs,
                             phase=phase, tileshape=tileshape,
                             transform_configurations=configs)    

    assert len(datasets) == 2
    # picard dataset should be acer, kirk dataset should be pinus
    picard, kirk = datasets
    assert picard.label == 0, 'picard is class acer and thus should have label 0'
    assert kirk.label == 1, 'kirk is class pinus and thus should have label 1'




@pytest.mark.parametrize(
        'name,transform,const',
        [('Zeroize', zeroize, torch.tensor(0.0)),
         ('Oneize', oneize, torch.tensor(1.0))]
)
def test_correct_workings_of_parametrized_transform(caplog, name, transform, const,
                                                    mock_instance_mapping, internal_path):
    
    caplog.set_level(logging.DEBUG)

    reader_class = ZarrReader

    # select a dataset instance via the test unique string ID
    picard = mock_instance_mapping['picard']
    path = picard.location
    classlabel_mapping = {
        'acer' : 1701,
        'pinus' : 2863
    }

    dataset = TileDataset(
        path=path,
        internal_path=internal_path,
        tileshape=(32, 32, 32),
        transformer=transform,
        classlabel_mapping=classlabel_mapping,
        reader_class=reader_class,
        phase='train'
    )

    # NOTE: torch is much more strict with data types than numpy, thus the convert
    # in conftest.py, we set the raw data of the Zarr array to be of type float32
    # if something fails here, it is likely due to the data type of the raw data
    const = const.to(torch.float32)
    expected_label = torch.tensor(1701)

    for data, label in dataset:
        assert torch.allclose(data, const)
        assert torch.isclose(label, expected_label)