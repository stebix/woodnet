import importlib
import torch
import logging
from pathlib import Path
from typing import Generator

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Type, Sequence

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)

# module filename prefix indicating custom contribution model module
PREFIX: str = 'customcontrib_'

ModelClass = Type[torch.nn.Module]


def collect_custom_model_modules(directory: Path) -> Generator[str, None, None]:
    """
    Collect custom contributed model modules via prefix-matching.
    """
    # we look for appropriately named modules in the models directory
    logger.debug(f'Collecting custom model modules from \'{directory}\'')
    for item in directory.iterdir():
        if item.is_file() and item.name.startswith(PREFIX):
            logger.debug(f'Found custom model module: {item}')
            yield item.stem


# this list holds the canonical model modules from which we can import via class name
# through the YAML configuration workflow
CANONICAL_MODEL_MODULES = [
    'woodnet.models.planar', 'woodnet.models.volumetric',
    *[
        f'woodnet.models.{item}'
        for item in list(collect_custom_model_modules(Path(__file__).parent))
    ]
]


def get_model_class(configuration: dict,
                    modules: Sequence[str] = CANONICAL_MODEL_MODULES) -> ModelClass:
    """
    Retrieve model class object by via string class name from the model config.
    Modifies model dictionary in-place. 

    Parameters
    ----------

    configuration : dict
        Model configuration dictionary. Must contain a 'name' key.

    modules : Sequence[str]
        Sequence of module names to search for the model class.


    Returns
    -------

    model_class : ModelClass
        Model class object.
    """
    name: str = configuration.pop('name')
    for module_name in modules:
        module = importlib.import_module(module_name)
        try:
            model_class = getattr(module, name)
            logger.debug(f'Retrieved model class {model_class} from module {module_name}')
            return model_class
        except AttributeError:
            pass
    raise ValueError(f'unrecognized model name \'{name}\'')



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
