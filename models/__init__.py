import importlib
import torch

from copy import deepcopy
from typing import Type


def get_model_class(configuration: dict) -> Type[torch.nn.Module]:
    """Retrieve model class by inferring name from the model config.
    
    Modifies model dictionary in-place. 
    """
    name: str = configuration.pop('name')
    module_names = ['models.planar', 'models.volume']
    for name in module_names:
        module = importlib.import_module(name)
        try:
            return getattr(module, name)
        except AttributeError:
            pass
    raise ValueError(f'unrecognized mode name \'{name}\'')



def create_model(configuration: dict) -> torch.nn.Module:
    """
    Create raw model object from top-level configuration.
    """
    configuration =  deepcopy(configuration)
    model_configuration = configuration.pop('model')
    model_class = get_model_class(model_configuration)
    return model_class(**model_configuration)