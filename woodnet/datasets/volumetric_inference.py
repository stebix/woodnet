import logging
import torch

from pathlib import Path
from typing import Any, Literal, Union
from collections.abc import Callable, Iterable

from torch import Tensor

from woodnet.datasets.volumetric import BaseTileDatasetBuilder, TileDataset
from woodnet.inference.parametrized_transforms import ParametrizedTransform

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


class ParametrizedTransformMixin:
    """
    Provide facilities for changeable parametrized transforms that modify the
    dataset output.
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



class TransformedTileDataset(TileDataset, ParametrizedTransformMixin):
    """
    TileDataset specialized for inference-time data-transform robustness experiments.

    The idea is that for inference-time robustness experiments, we provide facilities
    to include swappable parametrized transforms that modify the data.
    Compared to the standard transformer, these parametrized transforms are expected to be
    change many times over the dataset lifetime since we want to measure the model
    robustness for variable-strength transforms.
    """
    def __init__(self,
                 path: str | Path,
                 tileshape: tuple[int],
                 transformer: Callable[..., Any] | None = None,
                 parametrized_transform: Callable[[Tensor], Tensor] | None = None,
                 classlabel_mapping: dict[str, int] | None = None,
                 internal_path: str = 'downsampled/half',
                 phase: Literal['val'] = 'val') -> None:
        
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
        if self.parametrized_transform:
            parametrized_transform_info = {
                'name' : self.parametrized_transform.name,
                'parameters' : self.parametrized_transform.parameters
            }
        else:
            parametrized_transform_info = None
        infos = ', '.join((
            f"path='{self.path}'", f"phase='{self.phase}'",
            f"baseshape={self.baseshape}", f"tileshape={self.tileshape}",
            f"classlabel_mapping={self.classlabel_mapping}",
            f"has_transformer={has_transformer}",
            f'parametrized_transform={parametrized_transform_info}'
        ))
        return infos



class TransformedTileDatasetBuilder(BaseTileDatasetBuilder):
    """Build a TransformedTileDataset without a parametrized transform. Add this externally."""
    def build(cls,
              instances_ID: Iterable[str],
              tileshape: tuple[int, int, int],
              transform_configurations: Iterable[dict] | None = None,
              parametrized_transform: ParametrizedTransform | None = None
              ) -> list[TransformedTileDataset]:
        return super().build(TransformedTileDataset, instances_ID, 'val',
                             tileshape, transform_configurations,
                             parametrized_transform=parametrized_transform)



DataProvider = Union[torch.utils.data.Dataset, torch.utils.data.ConcatDataset,
                     list[torch.utils.data.Dataset], tuple[torch.utils.data.Dataset]]


def set_parametrized_transform(data: DataProvider, /, transform: ParametrizedTransform) -> None:
    """
    In-place set the parametrized transform of the given data provider.
    The abstract data provider may be a single dataset, a composite
    `torch.utils.data.ConcatDataset` or a list/tuple of `torch.utils.data.Dataset`. 
    The underlying datasets shoudl possess the `ParametrizedTransformMixin`
    to understand the role of the parametrized transform.

    Parameters
    ----------

    data : DataProvider, i.e. Dataset, ConcatDataset or list/tuple of Dataset
        Source data on which the parametrized_transform attribute will
        be set.
    
    transform : ParametrizedTransform
        Parametrized transformation globally applied on the dataset.

    Returns
    -------

    None
        Data(sets) are modified in-place.
    """
    if isinstance(data, torch.utils.data.ConcatDataset):
        for subset in data.datasets:
            subset.parametrized_transform = transform
    elif isinstance(data, (list, tuple)):
        for element in data:
            element.parametrized_transform = transform
    # TODO: this breaks probably when the transformed dataset hierarchy is extended
    # to the other datasets.
    elif isinstance(data, torch.utils.data.Dataset):
        data.parametrized_transform = transform
    else:
        raise TypeError(f'cannot set parametrized transform on invalid type {type(data)}')
