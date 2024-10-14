import torch
from functools import partial
from typing import Callable, Mapping, Iterable, Any

from woodnet.custom.exceptions import ConfigurationError

Tensor = torch.Tensor

def create_gradclip_func(conf: Mapping) -> Callable[[Iterable[Tensor] | Tensor], Any]:
    """
    Generate the gradient clipping function from ``torch.nn`` by partial
    application of the kwargs from the specialized sub-configuration.

    Parameters
    ----------

    conf : Mapping
        Sub-configuration for the gradient clipping function.

    Returns
    -------

    Callable[Iterable[torch.Tensor] | torch.Tensor] :
        The gradient clipping function.  
    """
    try:
        name = conf.pop('name')
    except KeyError as e:
        raise ConfigurationError('gradient clipping subconfiguration '
                                 'must define a name') from e

    if name == 'grad_norm':
        func = torch.nn.utils.clip_grad_norm_
    elif name == 'grad_value':
        func = torch.nn.utils.clip_grad_value_
    else:
        raise ConfigurationError(f'Unknown gradient clipping function: {name}')
    
    return partial(func, **conf)