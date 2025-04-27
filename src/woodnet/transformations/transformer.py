"""
Implement interfacing container classes for the transformation pipelines.

Jannik Stebani 2023
"""
import logging
import torch

from collections.abc import Callable

from woodnet.inference.parametrized_transforms import ParametrizedTransform

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


    def __str__(self) -> str:
        info_str = ''.join((self.__class__.__name__, '('))
        info_str += f'N={len(self.transforms)}'
        info_str += f'parametrized_transform={self.parametrized_transform}'
        return ''.join((info_str, ')'))
        

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


class ScriptedTransformer:

    def __init__(self, *transforms: torch.nn.Module) -> None:
        self.transforms = torch.nn.Sequential(*transforms)
        self.scripted_transforms = torch.jit.script(self.transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scripted_transforms(x)