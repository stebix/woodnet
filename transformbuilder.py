import importlib
import torch

from copy import deepcopy
from typing import Type, Iterable

from custom.exceptions import ConfigurationError


def get_class(name: str) -> Type[torch.nn.Module]:
    modules = [
        importlib.import_module(module_name)
        for module_name in ('torchvision.transforms', 'transform3D')
    ]
    for module in modules:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise ConfigurationError(f'could not find class with name "{name}"')


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