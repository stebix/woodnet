import importlib
import torch
from typing import Type

from custom.exceptions import ConfigurationError


def get_class(name: str) -> Type[torch.nn.Module]:
    """
    Get class object for a transformation by its string name.
    """
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


class Transformer:

    def __init__(self, *transforms: torch.nn.Module) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x
    
    def __str__(self) -> str:
        info_str = ''.join((self.__class__.__name__, '('))
        info_str += f'N={len(self.transforms)}'
        return ''.join((info_str, ')'))
        


class ScriptedTransformer:

    def __init__(self, *transforms: torch.nn.Module) -> None:
        self.transforms = torch.nn.Sequential(*transforms)
        self.scripted_transforms = torch.jit.script(self.transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scripted_transforms(x)