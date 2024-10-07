import torch
import pytest

from woodnet.models import create_model


def test_create_simple_resnet():
    config = {
        'model' : {
            'name' : 'ResNet18',
            'in_channels' : 2
        }
    }
    model = create_model(config)
    assert isinstance(model, torch.nn.Module), 'expecting torch module'


def test_create_3D_resnet():
    config = {
        'model' : {
            'name' : 'ResNet3D',
            'in_channels' : 1
        }
    }
    model = create_model(config)
    assert isinstance(model, torch.nn.Module), 'expecting torch module'

@pytest.mark.slowish
def test_compile_simple_resnet():
    config = {
        'model' : {
            'name' : 'ResNet18',
            'in_channels' : 1,
            'compile' : {
                'enabled' : True,
                'dynamic' : False,
                'fullgraph' : False
            }
        }
    }
    data = torch.randn(size=(1, 1, 256, 256))
    model = create_model(config)
    assert isinstance(model, torch.nn.Module), 'expecting torch module'
    _ = model(data)

