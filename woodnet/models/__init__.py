import importlib
import torch
import logging

from collections.abc import Callable, Mapping
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



def create_model(configuration: Mapping,
                 no_compile_override: bool = False) -> Callable:
    """
    Create raw model object from top-level configuration.

    Parameters
    ----------

    configuration : Mapping
        Top-level configuration. Must at least define a
        model subsection.

    no_compile_override : bool, optional
        Optional switch to fully disable compilation. Overrides
        any compilation options defined by the configuration mapping.
        Defaults to 'False', i.e. usage of the configuration options.

    
    Returns
    -------

    model : Callable (torch.nn.Module or OptimizedModule)
        Model instance. May be optmized/compiled
        depending on settings.
    """
    configuration =  deepcopy(configuration)
    model_configuration = configuration.pop('model')
    model_class = get_model_class(model_configuration)
    # determine compilation settings by user
    compile_configuration = model_configuration.pop('compile', {})
    config_compile_flag = compile_configuration.pop('enabled', False)

    model = model_class(**model_configuration)

    # This is the never compile branch
    if no_compile_override:
        if not config_compile_flag or not compile_configuration:
            message = f'Successfully created eager model object of {model_class}'
        if config_compile_flag:
            message = (f'Configuration-requested compilation is disabled via override flag. '
                       f'Successfully created eager model object of {model_class}')
        logger.info(message)
        return model

    if config_compile_flag:
        model = torch.compile(model, **compile_configuration)
        logger.info(f'Successfully created model object of {model_class} and '
                    f'compiled with options: {compile_configuration}')
        return model
    
    # basal fallthrough case: eager model is returned
    logger.info(f'Successfully created eager model object of {model_class}')
    return model
