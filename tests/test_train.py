import logging
import torch

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
def test_run_training_experiment_smoke():
    p = '/home/jannik/code/woodnet/woodnet/trainconf.yaml'
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


# TODO: reliant on local filesystem
def test_create_loaders():
    yaml = YAML(typ='safe')
    with open('/home/jannik/code/woodnet/woodnet/trainconf.yaml') as handle:
        conf = yaml.load(handle)

    loaders = create_loaders(conf)

    print(loaders)



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
