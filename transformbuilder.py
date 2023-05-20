import importlib
import torch

from copy import deepcopy
from typing import Type, Iterable

from custom.exceptions import ConfigurationError


def get_class(name: str) -> Type[torch.nn.Module]:
    module = importlib.import_module(
        'torchvision.transforms'
    )
    return getattr(module, name)


def from_configuration(cfg: dict, /) -> torch.nn.Module:
    """
    Create a transformation (callable `torch.nn.Module`) from
    a plain python configuration dictionary.
    """
    cfg = deepcopy(cfg)
    try:
        name = cfg.pop('name')
    except KeyError as e:
        raise ConfigurationError(
            'Configuration for automated transform building '
            'lacks the required "name" key'
        ) from e
    class_ = get_class(name)
    return class_(**cfg)


def from_configurations(cfgs: Iterable[dict]) -> list[torch.nn.Module]:
    """
    Create a list of transformations from an iterable of plain python
    configuration dictionaries.
    """
    return [from_configuration(cfg) for cfg in cfgs]