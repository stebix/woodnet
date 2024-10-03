from pathlib import Path

import pytest

from woodnet.datasets.triaxial import TriaxialDatasetBuilder



def test_builder(mock_instance_mapping, internal_path):
    # patch builder for local test-wise data configuration
    TriaxialDatasetBuilder.instance_mapping = mock_instance_mapping
    TriaxialDatasetBuilder.internal_path = internal_path
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
def test_provides_correct_orthoplane_shape(mock_instance_mapping, internal_path, tileshape):
    # patch builder for local test-wise data configuration
    TriaxialDatasetBuilder.instance_mapping = mock_instance_mapping
    TriaxialDatasetBuilder.internal_path = internal_path
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
    