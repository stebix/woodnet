import uuid
from pathlib import Path
from random import choice

import torch
import pytest

from woodnet.models import create_model
from woodnet.inference.resurrection import LazyModelDict


# number of strID to model path relations contained in the
# models mapping fixture
N_MODELS: int = 5
SUFFIX: str = 'pth'

@pytest.fixture(scope='module')
def configuration() -> dict:
    """Mock top-level configuration defining the model part."""
    modelconf = {
        'name' : 'ResNet3D',
        'in_channels' : 1
    }
    conf = {
        'model' : modelconf
    }
    return conf


@pytest.fixture(scope='module')
def models_mapping(configuration, tmp_path_factory) -> dict[str, Path]:
    """Create string ID to model parameters path on file system."""
    tmp_dir = tmp_path_factory.mktemp('models-mapping-dir')
    mapping = {}
    for _ in range(N_MODELS):
        model = create_model(configuration)
        ID = str(uuid.uuid4())
        savepath = tmp_dir / f'{ID}.{SUFFIX}'
        torch.save(model.state_dict(), savepath)
        mapping[ID] = savepath
    return mapping


def test_valid_insertion(configuration, models_mapping):
    models = LazyModelDict(configuration=configuration)
    for ID, modelpath in models_mapping.items():
        models[ID] = modelpath
    
    assert len(models) == N_MODELS


def test_valid_retrieval(configuration, models_mapping):
    """
    Test that a retrieved model instance is usable, i.e. 
    callable and a ResNet3D instanc.
    """
    # only required here for checking the return type of the model instance
    from woodnet.models.volumetric import ResNet3D
    # Deep learning settings
    dtype = torch.float32
    BATCH_SIZE: int = 2

    models = LazyModelDict(models_mapping, configuration=configuration)
    # randomly select one of the elements
    key = choice(list(models_mapping.keys()))
    model = models[key]

    assert isinstance(model, ResNet3D)

    data = torch.randn(BATCH_SIZE, 1, 64, 64, 64).to(dtype=dtype)
    model = model.to(dtype=dtype)
    out = model(data)
    assert isinstance(out, torch.Tensor)


@pytest.mark.slow
def test_kv_retrieval_via_items_method(configuration, models_mapping):
    """
    Test that the iterator over key - value pairs via the `items()` method 
    is usable and correct, i.e. that string ID and callable ResNet3D instances
    are returned.
    """
    # only required here for checking the return type of the model instance
    from woodnet.models.volumetric import ResNet3D
    # Deep learning settings
    dtype = torch.float32
    BATCH_SIZE: int = 2
    data = torch.randn(BATCH_SIZE, 1, 64, 64, 64).to(dtype=dtype)


    models = LazyModelDict(models_mapping, configuration=configuration)
    # iterator over key - value pairs
    for ID, model in models.items():
        assert isinstance(ID, str)
        assert isinstance(model, ResNet3D)

        model = model.to(dtype)
        out = model(data)
        assert isinstance(out, torch.Tensor)



    










