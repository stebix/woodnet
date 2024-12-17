import logging
import torch
import string
import numpy as np

import pytest
from ruamel.yaml import YAML

import woodnet.logtools.dict.ops as logged
from woodnet.models import create_model
from woodnet.models.volumetric import ResNet3D
from woodnet.custom.exceptions import ConfigurationError
# weird double import as we need to access the trainmodule to monkeypatch a module-level variable
import woodnet.train as trainmodule
from woodnet.train import (create_optimizer, create_loss, create_loaders,
                           run_training_experiment, get_ID_overlap, check_ID_overlap)

import woodnet.datasets.setup as setup
import woodnet.datasets.volumetric

def test_retrieve_logged(caplog):
    caplog.set_level(logging.DEBUG)
    data = {
        'setting' : True,
        'parameter' : 'warp 9'
    }
    value_a = logged.retrieve(data, key='energize', default='do not beam me up', method='get')
    value_b = logged.retrieve(data, key='parameter', default='warp 1', method='get')

    assert value_a == 'do not beam me up'
    assert value_b == 'warp 9'



# TODO: reliant on local filesystem
@pytest.mark.skip
def test_run_training_experiment_smoke():
    p = '/home/jannik/code/woodnet/tests/assets/trainconf.yaml'
    run_training_experiment(p)


def test_train_smoke():
    config = """
    model:
        name: ResNet3D
        in_channels: 1
        compile:
            enabled: False
            dynamic: False
            fullgraph: False

    optimizer:
        name: Adam
        learning_rate: 1e-3
    """
    yaml = YAML(typ='safe')
    config: dict = yaml.load(config)

    model = create_model(config)
    opt = create_optimizer(model, config)

    assert isinstance(model, ResNet3D)
    assert isinstance(opt, torch.optim.Adam)



def test_create_loss():
    config = """
    loss:
        name: 'BCEWithLogitsLoss'
        reduction: 'mean'
    """
    yaml = YAML(typ='safe')
    config: dict = yaml.load(config)
    loss = create_loss(config)
    assert isinstance(loss, torch.nn.BCEWithLogitsLoss)


def test_create_loaders(synthetic_dataset, monkeypatch):
    tileshape: tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 2
    instances_per_phase: int = 2
    expected_output_size = (batch_size, 1, *tileshape)
    instance_mapping = synthetic_dataset.instance_mapping
    # We need to patch this since the instance mapping variable
    # for the builder class is set at import-time
    monkeypatch.setattr(setup, 'INSTANCE_MAPPING', value=instance_mapping)
    monkeypatch.setattr(
        woodnet.datasets.volumetric.BaseTileDatasetBuilder,
        'instance_mapping',
        value=instance_mapping
    )
    monkeypatch.setattr(
        woodnet.datasets.volumetric.BaseTileDatasetBuilder,
        'internal_path',
        value=synthetic_dataset.internal_path
    )
    monkeypatch.setattr(
        woodnet.datasets.volumetric.BaseTileDatasetBuilder,
        'classlabel_mapping',
        value=synthetic_dataset.classlabel_mapping
    )
    raw_configuration = """
        loaders:
            dataset: TileDataset
            tileshape: [ $TILESHAPE ]
            batchsize: $BATCH_SIZE
            num_workers: 0
            pin_memory: True

            train:
                instances_ID: [ $TRAINING_ID ]

                transform_configurations:
                    - name: Normalize
                      mean: 110
                      std: 950
        
            val:
                instances_ID: [ $VALIDATION_ID ]

                transform_configurations:
                    - name: Normalize
                      mean: 110
                      std: 950
    """
    configuration = string.Template(raw_configuration)

    # Generically select IDs and stuff them into the configuration template.
    rng = np.random.default_rng()
    IDs = np.array(list(instance_mapping.keys()))
    train_ID, val_ID = rng.choice(IDs, size=(2, instances_per_phase), replace=False)
    train_ID = ', '.join((str(c) for c in train_ID))
    val_ID = ', '.join((str(c) for c in val_ID))

    tileshape_str = ', '.join((str(d) for d in tileshape))

    configuration = configuration.substitute(
        {'TRAINING_ID': train_ID, 'VALIDATION_ID': val_ID,
         'TILESHAPE' : tileshape_str, 'BATCH_SIZE' : str(batch_size)}
    )
    # Load config realistically via YAML parser.
    yaml = YAML(typ='safe')
    conf = yaml.load(configuration)

    import rich
    rich.print(configuration)
    rich.print(synthetic_dataset.instance_mapping)

    rich.print(IDs)
    rich.print(train_ID)


    # Core test.
    loaders = create_loaders(conf)
    
    trainloader = loaders['train']
    valloader = loaders['val']
    train_data, train_label = next(iter(trainloader))
    val_data, val_label = next(iter(valloader))

    # Shape of the data should be of the format (N, C, D, H, W)
    assert train_data.shape == expected_output_size, (
        f'train loader returned {train_data.shape} '
        f'sized tensor, expected {expected_output_size}')
    
    assert val_data.shape == expected_output_size, (
        f'val loader returned {val_data.shape} sized tensor, '
        f'expected {expected_output_size}')



def test_get_ID_overlap_for_no_overlap():
    phase_configs = {
        'train' : {
            'instances_ID' : ['1', '2', '3']
        },
        'val' : {
            'instances_ID' : ['4', '5', '6']
        }
    }
    overlap = get_ID_overlap(phase_configs)
    assert not overlap



def test_get_ID_overlap_for_overlap_of_length_one():
    phase_configs = {
        'train' : {
            'instances_ID' : ['1', '2', '3']
        },
        'val' : {
            'instances_ID' : ['4', '5', '6', '3']
        }
    }
    overlap = get_ID_overlap(phase_configs)
    assert len(overlap) == 1
    assert overlap == set(['3'])



def test_get_ID_overlap_for_large_overlaps():
    phase_configs = {
        'train' : {
            'instances_ID' : ['1', '2', '3', '4']
        },
        'val' : {
            'instances_ID' : ['4', '5', '6', '3']
        },
        'test' : {
            'instances_ID' : ['3', '4', '5', '6']
        }
    }
    overlap = get_ID_overlap(phase_configs)
    assert len(overlap) == 4


def test_check_ID_overlap_warns_for_actual_overlap(monkeypatch):
    monkeypatch.setattr(target=trainmodule, name='TRAIN_VAL_OVERLAP_ACTION', value='warn')
    phase_configs = {
        'train' : {
            'instances_ID' : ['1', '2', '3']
        },
        'val' : {
            'instances_ID' : ['4', '5', '6', '3']
        }
    }
    with pytest.warns(UserWarning, match='Overlapping IDs'):
        check_ID_overlap(phase_configs)


def test_check_ID_overlap_raises_for_actual_overlap(monkeypatch):
    monkeypatch.setattr(target=trainmodule, name='TRAIN_VAL_OVERLAP_ACTION', value='raise')
    phase_configs = {
        'train' : {
            'instances_ID' : ['1', '2', '3']
        },
        'val' : {
            'instances_ID' : ['4', '5', '6', '3']
        }
    }
    with pytest.raises(ConfigurationError):
        check_ID_overlap(phase_configs)
