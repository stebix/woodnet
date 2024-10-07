"""
Tooling for automated resurrection, i.e. model instance creation and trained parameters loading.

@jsteb 2024
"""
import logging
from collections import OrderedDict, UserDict
from collections.abc import Mapping, Callable
from pathlib import Path

import torch

from woodnet.models import create_model

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)



def configure_model(model: torch.nn.Module,
                    dtype: torch.dtype,
                    device: torch.device,
                    eval_mode: bool,
                    testing_flag: bool) -> torch.nn.Module:
    """
    Configure model in-place according to settings.
    
    Parameters
    ----------

    model : torch.nn.Module
        Model instance to be configured.

    dtype : torch.dtype
        Data type to be used for model parameters.

    device : torch.device
        Device to be used for model. Weights are pushed to this device.

    eval_mode : bool
        Flag to set model to evaluation mode.

    testing_flag : bool
        Flag to set model to testing mode.
        Usually this setting triggers the application of the final nonlinearity.
        For testing, the nonlinearity is applied.
        For training (non-testing), the nonlinearity is usually absorbed into the loss function.

    Returns
    -------

    model : torch.nn.Module
    """
    model = model.to(dtype=dtype, device=device)
    if eval_mode:
        model.eval()
    if testing_flag:
        model.testing = True
    return model


def transmogrify_state_dict(state_dict: Mapping) -> OrderedDict:
    """
    Try to turn state dict from compiled model into normal state dict.
    For `torch.compile` models, the state dict is prefixed with '_orig_mod.'

    Parameters
    ----------

    state_dict : Mapping
        State dictionary from compiled model.

    Returns
    -------

    state_dict : OrderedDict
        State dictionary with prefix removed.
    """
    prefix: str = '_orig_mod.'
    return OrderedDict({key.removeprefix(prefix) : value for key, value in state_dict.items()})


def inject_state_dict(model: torch.nn.Module, state_dict: Mapping) -> None:
    """
    Loads state dict into the given model in-place.
    Helper function that automates the handling of state dicts reconstructed
    from compiled models via a transmogrification attempt.

    Parameters
    ----------

    model : torch.nn.Module
        Model instance to be loaded with state dict.

    state_dict : Mapping
        State dictionary to be loaded into the model. Can originate from
        an uncompiled or compiled model.

    """
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        logger.warning('Direct state dict loading failed with runtime error. Re-attempting '
                       'with transmogrified state dict.')
        state_dict = transmogrify_state_dict(state_dict)
        model.load_state_dict(state_dict)
    logger.info('Successfully loaded state dictionary.')



class LazyModelDict(UserDict):
    """
    Dict-like container that resurrects models lazily upon request via key.
    
    The model is prouced from the basic ingredients.

    Basic ingredients:
        - Mapping from unique model string ID to checkpoint file location.
        - (top-level) configuration containing the model specification
          e.g. name, kwargs and fully optional compile options
    
    Model object will be newly created from ground up upon value retrieval.
    """
    # Configuration is required but kwarg, maybe design this better.
    def __init__(self, dict=None, /, configuration: Mapping = None, no_compile_override: bool = False) -> None:
        if configuration is None:
            raise TypeError('LazyModelDict requires configuration')
        self._configuration = configuration
        self.no_compile_override = no_compile_override
        super().__init__(dict)
    

    def __setitem__(self, key: str, item: Path) -> None:
        if not isinstance(item, Path):
            logger.warning(f'LazyModelDict expects pathlib.Path values, but got {type(item)}')
        try:
            suffix = item.suffix
        except AttributeError:
            suffix = 'NOTSET'
        
        if not suffix.endswith('pth'):
            logger.warning('Inserted value does have expected \'pth\' suffix.')

        return super().__setitem__(key, item)
    

    def __getitem__(self, key: str) -> Callable | torch.nn.Module:
        """
        Retrieve model item via string ID key.
        Model will be lazily instantiated on the fly.

        NOTE: Currently we create the randomly-init'ed model, compile it and then
              load the trained parameters via the state_dict.
              This may have unforeseen performance consequences. Check this!
        """
        path = super().__getitem__(key)
        logger.debug(f'Requested model instance (ID=\'{key}\') from location \'{path}\'')
        model = create_model(self._configuration, no_compile_override=self.no_compile_override)
        # first force load to CPU/RAM: training and inference device may differ
        state_dict = torch.load(path, map_location='cpu')
        inject_state_dict(model, state_dict)
        logger.debug(f'Successfully re-created model from location: \'{path}\'.')
        return model
        

def resurrect_models_from(pathmap: Mapping[str, Path],
                          configuration: Mapping,
                          dtype: torch.dtype,
                          device: torch.device,
                          no_compile_override: bool = False,
                          eval_mode: bool = True,
                          testing_flag: bool = True
                          ) -> dict[str, Callable | torch.nn.Module]:
    """
    Eagerly resurrect models by constructing a mapping from string ID to
    callable model instance from a path mapping.

    Parameters
    ----------

    pathmap : Mapping[str, Path]
        Mapping from unique model string ID to checkpoint file location.

    configuration : Mapping
        Top-level configuration. Must at least define a
        model subsection.

    dtype : torch.dtype
        Data type to be used for model parameters.

    device : torch.device
        Device to be used for model. Weights are pushed to this device.

    no_compile_override : bool, optional
        Optional switch to fully disable compilation. Overrides
        any compilation options defined by the configuration mapping.
        Defaults to 'False', i.e. usage of the configuration options.

    eval_mode : bool, optional
        Flag to set model to evaluation mode. Defaults to 'True'.

    testing_flag : bool, optional
        Flag to set model to testing mode. Defaults to 'True'.
        Usually this setting triggers the application of the final nonlinearity.
        For testing, the nonlinearity is applied.
        For training (non-testing), the nonlinearity is usually absorbed into the loss function.

        
    Returns
    -------

    models : dict[str, Callable | torch.nn.Module]
        Mapping from string ID to model instances that are fully configured
        according to input settings.
    """
    models = {}
    for ID, path in pathmap.items():
        logger.debug(f'Starting creation of model instance (ID=\'{ID}\''
                     f') from location \'{path}\'')
        model = create_model(configuration, no_compile_override=no_compile_override)
        state_dict = torch.load(path, map_location='cpu')
        inject_state_dict(model, state_dict)
        model = configure_model(model, dtype=dtype, device=device, eval_mode=eval_mode,
                                testing_flag=testing_flag)
        logger.debug(f'Successfully re-created and configured model.')
        models[ID] = model
    return models


        
