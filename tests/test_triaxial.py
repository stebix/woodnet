from pathlib import Path

import numpy as np
import pytest
import zarr
import zarr.convenience

from woodnet.datasets.triaxial import TriaxialDatasetBuilder
from woodnet.datasets.setup import InstanceFingerprint


BASE_DATA_SHAPE: tuple[int, int, int, int] = (1, 128, 128, 128)
INTERNAL_PATH: str = 'group/data'


@pytest.fixture(scope='module')
def tempdir(tmp_path_factory):
    return tmp_path_factory.mktemp('mock-raw-data')


def create_zarr_dataset(directory: Path, classname: str) -> Path:
    """
    Create an authentic zarr dataset with a fingerprint inside the
    given directory. The dataset will be created with a random
    data array of shape BASE_DATA_SHAPE.
    """
    data = np.random.default_rng().normal(0, 1, size=BASE_DATA_SHAPE)
    fingerprint = {
        'class_': classname,
        'voltage': '40kV',
        'current': '200muA',
        'duration': '1s',
        'averages': 2
    }
    zarr_path = directory / f'{classname}-testdataset.zarr'
    group, dataset = INTERNAL_PATH.split('/')
    # create internal structure and write data
    array = zarr.open(zarr_path, mode='w')
    group = array.create_group(group)
    dataset = group.create_dataset(dataset, data=data)
    # write metadata fingerprint
    for k, v in fingerprint.items():
        array.attrs[k] = v
    
    return zarr_path



@pytest.fixture(scope='module')
def mock_instance_mapping(tempdir):
    class_a = 'acer'
    class_b = 'pinus'
    instance_mapping_raw = {
        'picard' : {
            'location' : str(create_zarr_dataset(tempdir, class_a)),
            'classname' : class_a,
            'group' : 'axial'
        },
        'kirk' : {
            'location' : str(create_zarr_dataset(tempdir, class_b)),
            'classname' : class_b,
            'group' : 'tangential'
        }
    }
    instance_mapping = {k : InstanceFingerprint(**v) for k, v in instance_mapping_raw.items()}
    return instance_mapping


def test_builder(mock_instance_mapping):
    # patch builder for local test-wise data configuration
    TriaxialDatasetBuilder.instance_mapping = mock_instance_mapping
    TriaxialDatasetBuilder.internal_path = INTERNAL_PATH
    TriaxialDatasetBuilder.classlabel_mapping = {
        'acer' : 1701,
        'pinus' : 2893
    }

    builder = TriaxialDatasetBuilder()
    datasets = builder.build(
        instances_ID=['picard', 'kirk'],
        phase='val',
        tileshape=(64, 64, 64),
        planestride=(32, 32, 32),
        transform_configurations=None
    )
    assert len(datasets) == 2
    
    picard, kirk = datasets
    assert picard.label == 1701, 'picard is class acer and thus should have label 1701'
    assert kirk.label == 2893, 'kirk is class pinus and thus should have label 2893'


@pytest.mark.parametrize('tileshape', [(64, 64, 64), (32, 32, 32)])
def test_provides_correct_orthoplane_shape(mock_instance_mapping, tileshape):
    # patch builder for local test-wise data configuration
    TriaxialDatasetBuilder.instance_mapping = mock_instance_mapping
    TriaxialDatasetBuilder.internal_path = INTERNAL_PATH
    TriaxialDatasetBuilder.classlabel_mapping = {
        'acer' : 1701,
        'pinus' : 2893
    }
    builder = TriaxialDatasetBuilder()
    datasets = builder.build(
        instances_ID=['picard'],
        phase='train',
        tileshape=tileshape,
        planestride=(32, 32, 32),
        transform_configurations=None
    )
    dataset = datasets[0]
    element, label = dataset[0]
    assert label == 1701, 'picard is class acer and thus should have label 1701'
    _, *planeshape = element.shape
    assert element.shape == (3, *planeshape), f'orthoplane shape (3, ty, tx) but got {element.shape}'
    