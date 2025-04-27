"""
Implement interfacing container classes for the transformation pipelines.

Jannik Stebani 2023
"""
import logging
import torch

from collections.abc import Callable, Mapping, Sequence

from woodnet.inference.parametrized_transforms import ParametrizedTransform
from woodnet.transformations.buildtools import from_configurations

TensorTransform = Callable[[torch.Tensor], torch.Tensor] | torch.nn.Module

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


class Transformer:

    def __init__(self, *transforms: TensorTransform,
                 parametrized_transform: ParametrizedTransform | None = None) -> None:
        # TODO: Basic Python question: does * provide us with a list or a tuple???
        self.transforms = list(transforms)
        self._parametrized_transform = None
        self.parametrized_transform = parametrized_transform


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)

        if self.parametrized_transform:
            x = self.parametrized_transform.transform(x)

        return x


    @property
    def parametrized_transform(self) -> ParametrizedTransform:
        return self._parametrized_transform


    @parametrized_transform.setter
    def parametrized_transform(self, new: ParametrizedTransform | None) -> None:
        """Set a new parameterized transform and emit log message."""
        if new is None:
            logger.info('Disabled parametrized transform via <None> value.')
            self._parametrized_transform = new
            return

        message = f'{self.__class__.__name__} received new parametrized transform: {new}.'
        level = logging.DEBUG

        if not isinstance(new, ParametrizedTransform):
            level = logging.WARNING
            addendum = f'Expected callable of {ParametrizedTransform}, but got {type(new)}!'
            message = ' '.join((message, addendum))
        
        logger.log(level, message)
        self._parametrized_transform = new


    def append(self, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Append a new transform to the transformation pipeline."""
        if not callable(transform):
            raise TypeError(f'Expected callable, but got {type(transform)}!')
        self.transforms.append(transform)
        logger.debug(f'Appended new transform: {transform}.')


    def insert(self, index: int, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Insert a new transform at the given index."""
        if not callable(transform):
            raise TypeError(f'Expected callable, but got {type(transform)}!')
        self.transforms.insert(index, transform)
        logger.debug(f'Inserted new transform: {transform} at index {index}.')


    def prepend(self, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Prepend a new transform to the transformation pipeline."""
        if not callable(transform):
            raise TypeError(f'Expected callable, but got {type(transform)}!')
        self.transforms.insert(0, transform)
        logger.debug(f'Prepended new transform: {transform}.')


    @classmethod
    def from_configurations(cls, configurations: Sequence[Mapping]) -> 'Transformer':
        """
        Create a Transformer instance from a list of configuration
        dictionaries.
        """
        if configurations is None:
            configurations = []
        transforms = from_configurations(configurations)
        return cls(*transforms)


    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        format_string += '(\n'
        format_string += f'  parametrized_transform={self.parametrized_transform},\n'
        format_string +=  '  transforms=\n    ['
        for t in self.transforms:
            format_string += '\n'
            format_string += f'        {t}'
        format_string += '\n    ]\n)'
        return format_string


    def __str__(self) -> str:
        return repr(self)


class ScriptedTransformer:

    def __init__(self, *transforms: torch.nn.Module) -> None:
        self.transforms = torch.nn.Sequential(*transforms)
        self.scripted_transforms = torch.jit.script(self.transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scripted_transforms(x)