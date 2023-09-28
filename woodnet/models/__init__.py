import importlib
import torch
import logging

from collections.abc import Callable
from copy import deepcopy
from typing import Type

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def get_model_class(configuration: dict) -> Type[torch.nn.Module]:
    """Retrieve model class by inferring name from the model config.
    
    Modifies model dictionary in-place. 
    """
    name: str = configuration.pop('name')
    module_names = ['woodnet.models.planar', 'woodnet.models.volumetric']
    for module_name in module_names:
        module = importlib.import_module(module_name)
        try:
            return getattr(module, name)
        except AttributeError:
            pass
    raise ValueError(f'unrecognized mode name \'{name}\'')



def create_model(configuration: dict) -> Callable:
    """
    Create raw model object from top-level configuration.
    """
    configuration =  deepcopy(configuration)
    model_configuration = configuration.pop('model')
    model_class = get_model_class(model_configuration)
    # determine compilation settings by user
    compile_configuration = model_configuration.pop('compile', {})
    compile_flag = compile_configuration.pop('enabled', False)

    model = model_class(**model_configuration) 

    if not compile_configuration or not compile_flag:
        logger.debug('Successfully created eager model object')
        return model
    else:
        model = torch.compile(model, **compile_configuration)
        logger.debug(f'Successfully compiled model object with options: {compile_configuration}')
        return model
