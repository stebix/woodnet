"""
Container encapsulating transform instances as their members.

@jsteb 2023
"""
import random
import torch

from copy import deepcopy
from collections.abc import Sequence, Callable

from woodnet.transformations.buildtools import from_configuration

Tensor = torch.Tensor

class EquiprobableSelector:
    """
    Container that can hold any number of internals transforms.
    Selects one of the internal transforms upon being called, each one
    with equal probability and applies it to the provided input.
    """
    propagate_seed: bool = False

    def __init__(self, members: Sequence[dict], seed: int = 123, **kwargs) -> None:
        self._seed = seed
        self._transforms = self.create_transforms(members=members)
        self._kwargs = kwargs
        

    def __call__(self, tensor: Tensor) -> Tensor:
        transform = random.choice(self._transforms)
        return transform(tensor)
    

    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        format_string += f'(random_seed={self._random_seed}, ['
        for t in self._transforms:
            format_string += '\n'
            format_string += f'       {t}'
        format_string += ']\n)'
        return format_string


    def create_transforms(self, members: Sequence[dict]) -> list[Callable]:
        members = deepcopy(members)
        transforms = []
        for configuration in members:
            if self.propagate_seed:
                configuration.update({'seed' : self._seed})
            transforms.append(from_configuration(configuration))
        return transforms
            