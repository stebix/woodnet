import logging
import torch

from pathlib import Path
from typing import Any, NamedTuple
from collections.abc import Callable, Mapping

from torch import Tensor

from woodnet.datasets.volumetric import TileDataset


LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


class ParametrizedTransform(NamedTuple):
    name: str
    parameters: Mapping
    transform: Callable



class InferenceDatasetMixin:
    """
    Provide facilities for changeable parametrized transforms.
    """
    @property
    def parametrized_transform(self) -> ParametrizedTransform:
        return self._parametrized_transform


    @parametrized_transform.setter
    def parametrized_transform(self, new: ParametrizedTransform) -> None:
        """Set a new parameterized transform and emit log message."""
        message = f'{self.__class__.__name__} received new parametrized transform: {new}.'
        level = logging.DEBUG
        if not isinstance(new, ParametrizedTransform):
            level = logging.WARNING
            addendum = f'Transform is not instance of {type(ParametrizedTransform)}!'
            message = ' '.join((message, addendum))
        
        logger.log(level, message)
        self._parametrized_transform = new



class InferenceTileDataset(TileDataset, InferenceDatasetMixin):
    """
    TileDataset specialized for inference ablation experiments.

    In comparison to the bog-standard dataset, this class provides facilities to
    inject a `parametrized_transform` that globally modifies the output.
    """
    def __init__(self,
                 path: str | Path,
                 tileshape: tuple[int],
                 transformer: Callable[..., Any] | None = None,
                 parametrized_transform: Callable[[Tensor], Tensor] | None = None,
                 classlabel_mapping: dict[str, int] | None = None,
                 internal_path: str = 'downsampled/half') -> None:
        
        phase: str = 'val'
        super().__init__(path, phase, tileshape, transformer, classlabel_mapping, internal_path)
        self._parametrized_transform = parametrized_transform
        

    def __getitem__(self, index: int) -> tuple[Tensor] | Tensor:
        """
        Retrieve dataset item: tuple of tensor for training phase
        (data and label) or test phase (single tensor).
        """
        tile = self.tilebuilder.tiles[index]
        subvolume = self.volume[tile]
        subvolume = torch.tensor(subvolume)

        if self.transformer:
            subvolume = self.transformer(subvolume)
        
        if self.parametrized_transform:
            subvolume = self.parametrized_transform.transform(subvolume)

        if self.phase == 'test':
            return subvolume
        
        label = torch.tensor(self.label).unsqueeze_(-1)

        return (subvolume, label)
    

    def _make_info_str(self) -> str:
        has_transformer = True if self.transformer else False
        parametrized_transform = {
            'name' : self.parametrized_transform.name,
            'parameters' : self.parametrized_transform.parameters
        }
        infos = ', '.join((
            f"path='{self.path}'", f"phase='{self.phase}'",
            f"baseshape={self.baseshape}", f"tileshape={self.tileshape}",
            f"classlabel_mapping={self.classlabel_mapping}",
            f"has_transformer={has_transformer}",
            f'parametrized_transform={parametrized_transform}'
        ))
        return infos
