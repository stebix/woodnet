"""
Implement general interface to create transformations
acting upon data instances.

Jannik Stebani 2023
"""
import importlib
import torch

from collections.abc import Iterable
from copy import deepcopy
from typing import Type

from woodnet.custom.exceptions import ConfigurationError


def get_class(name: str) -> Type[torch.nn.Module]:
    """
    Get class object for a transformation by its string name.
    """
    modules = [
        importlib.import_module(module_name)
        for module_name in ('torchvision.transforms', 'woodnet.transformations.transforms')
    ]
    for module in modules:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise ConfigurationError(f'could not find class with name "{name}"')


def from_configuration(configuration: dict, /) -> torch.nn.Module:
    """
    Create a transformation (generally a callable `torch.nn.Module`) from
    a plain python configuration dictionary.
    """
    configuration = deepcopy(configuration)
    try:
        name = configuration.pop('name')
    except KeyError as e:
        raise ConfigurationError(
            'Configuration for automated transform building '
            'lacks the required "name" key'
        ) from e
    class_ = get_class(name)
    return class_(**configuration)


def from_configurations(configurations: Iterable[dict]) -> list[torch.nn.Module]:
    """
    Create a list of transformations from an iterable of plain python
    configuration dictionaries.
    """
    return [from_configuration(conf) for conf in configurations]