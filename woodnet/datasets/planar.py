"""
Implement 2D datasets for the wood CT data.

Jannik Stebani 2023
"""
import torch
import numpy as np
import logging

from collections.abc import Callable, Iterable, Mapping
from functools import cached_property
from torch import Tensor
from typing import Optional, Literal, Union, NamedTuple
from pathlib import Path

import tqdm.auto as tqdm
import zarr

from woodnet.datasets.utils import add_channel_dim
from woodnet.dataobjects import AbstractSlice, Volume
from woodnet.datasets.constants import DEFAULT_CLASSLABEL_MAPPING, CLASSNAME_REMAP
from woodnet.inference.parametrized_transforms import ParametrizedTransform
from woodnet.transformations.transformer import Transformer
from woodnet.transformations.buildtools import from_configurations 
from woodnet.datasets.tiling import PlanarTileBuilder


LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


class SliceDataset(torch.utils.data.Dataset):
    """
    Dataset consisting of many 2D slices (H x W)
    """
    def __init__(self,
                 phase: Literal['train', 'test'],
                 slices: list[AbstractSlice],
                 classlabel_mapping: dict[str, int] | None = None,
                 transformer: Callable | None = None,
                 ) -> None:
        
        self.phase = phase
        self.classlabel_mapping = classlabel_mapping
        self.transformer = transformer
        
        if self.phase == 'train':
            if self.classlabel_mapping is None:
                raise TypeError(
                    f'{self.__class__.__name__} requires class to label mapping '
                     'at for the selected train phase'
                )

        self.slices = slices


    def __getitem__(self, index: int) -> Union[Tensor, tuple[Tensor, int]]:
        """
        Return signature is phase-dependent:
            - train phase has tuple output
                 -> (input [Tensor, (H x W)], label [integer])
            - test phase has tensor output
                 ->  input [Tensor, (H x W)]
        """
        slc = self.slices[index]
        data = torch.tensor(add_channel_dim(slc.data))

        if self.transformer:
            data = self.transformer(data)

        if self.phase == 'test':
            return data

        label = torch.tensor(self.classlabel_mapping[slc.class_]).unsqueeze(-1)

        return (data, label.to(torch.long))
    

    def __len__(self) -> int:
        return len(self.slices)
        
        
        

class EagerSliceDataset(torch.utils.data.Dataset):
    """
    Eager version of the slice dataset that loads the full volume into main
    memory as numpy.ndarray.

    Useful for repeated prediction e.g. in the context of a 
    jupyter notebook environment.
    """
    # TODO: Refactor constructor without expicit dependency on volume
    def __init__(self,
                 phase: Literal['train', 'val'],
                 volume: np.ndarray,
                 fingerprint: Mapping,
                 transformer: Optional[callable] = None,
                 classlabel_mapping: Optional[dict[str, int]] = None,
                 axis: int = 0,
                 ) -> None:

        self.phase = phase
        self.axis = axis
        self.fingerprint = fingerprint
        self.volume = np.swapaxes(volume, 0, axis)
        self.transformer = transformer

        if self.phase in {'train', 'val'} and classlabel_mapping is None:
            raise RuntimeError(f'Phase \'{self.phase}\' dataset requires a '
                                'classlabel mapping!')

        self.classlabel_mapping = classlabel_mapping


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | Tensor:        
        data = torch.tensor(add_channel_dim(self.volume[index]))

        if self.transformer:
            data = self.transformer(data)

        if self.phase == 'test':
            return data
        
        label = torch.tensor(self.label).unsqueeze(-1)
        
        return (data, label)


    @cached_property
    def label(self) -> int:
        classname = self.fingerprint['class_']
        try:
            classvalue = self.classlabel_mapping[classname]
        except KeyError:
            classvalue = self.classlabel_mapping[CLASSNAME_REMAP[classname]]
        return classvalue


    def __len__(self) -> int:
        return self.volume.shape[0]
    


Tileshape2D = tuple[int, int]
Tileshape3D = tuple[int, int, int]
Tilespec2D = tuple[int, tuple[slice, ...]]
Tilespec3D = tuple[slice, ...]


class StagedIndex(NamedTuple):
    z_plane_index: int
    in_plane_index: int


def make_get_z_index(tiles_per_plane: int, z_size: int, z_spacing: int
                     ) -> tuple[int, Callable[[int], StagedIndex]]:
    """
    Creation routine for a z index getter function that provides a mapping from a global
    scalar instance index to the staged indices that holds the z plane index in the first
    element and the in-plane tile index in the second element.
    """
    z_planes: int = z_size // z_spacing
    index_intervals = [range(zi*tiles_per_plane, (zi+1)*tiles_per_plane) for zi in range(z_planes)]
    total_elements = tiles_per_plane * (z_size // z_spacing)
    
    def get_z_index(index: int) -> StagedIndex:
        for z_plane_index, interval in enumerate(index_intervals):
            if index in interval:
                in_plane_index = index - (z_plane_index * tiles_per_plane)
                return StagedIndex(z_plane_index=z_plane_index, in_plane_index=in_plane_index)
        raise IndexError(f'Index {index} out of bounds for setting: {tiles_per_plane} tiles per plane, '
                         f'total z size {z_size} and z spacing of {z_spacing}. '
                         f'Total expected elements: {total_elements}.')
        
    return (total_elements, get_z_index)


class TiledEagerSliceDataset(torch.utils.data.Dataset):
    """
    Eager version of the slice dataset that loads the full volume into main
    memory as numpy.ndarray.

    Useful for repeated prediction e.g. in the context of a 
    jupyter notebook environment.
    """
    # TODO: Refactor constructor without expicit dependency on volume
    def __init__(self,
                 phase: Literal['train', 'val'],
                 volume: np.ndarray,
                 fingerprint: Mapping,
                 tileshape: tuple[int, int],
                 z_spacing: int,
                 transformer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 classlabel_mapping: Optional[dict[str, int]] = None,
                 axis: int = 0,
                 ) -> None:

        self.phase = phase
        self.axis = axis
        self.fingerprint = fingerprint
        self.volume = np.swapaxes(volume, 0, axis)
        self.tileshape = tileshape
        self.transformer = transformer

        self.tileshape = tileshape
        self.z_spacing = z_spacing

        self.z_size = self.volume.shape[0]
        self.baseshape = self.volume.shape[1:]
        self.radius = self.baseshape[-1] // 2
        logger.info(f'Constructed {self.__class__.__name__} with deduced baseshape '
                    f'of {self.baseshape} and radius of {self.radius}')
        
        self.tilebuilder = PlanarTileBuilder(baseshape=self.baseshape,
                                             tileshape=tileshape,
                                             radius=self.radius)
        
        tiles_per_plane = len(self.tilebuilder.tiles)
        self.length, self.index_transformer = make_get_z_index(tiles_per_plane=tiles_per_plane,
                                                               z_size=self.z_size,
                                                               z_spacing=z_spacing)

        if self.phase in {'train', 'val'} and classlabel_mapping is None:
            raise RuntimeError(f'Phase \'{self.phase}\' dataset requires a '
                                'classlabel mapping!')

        self.classlabel_mapping = classlabel_mapping


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | Tensor:        
        # transform flat index into staged/hierarchical index
        staged_index = self.index_transformer(index)
        tile = self.tilebuilder.tiles[staged_index.in_plane_index]
        data = torch.tensor(add_channel_dim(
            self.volume[staged_index.z_plane_index][tile])
        )

        if self.transformer:
            data = self.transformer(data)

        if self.phase == 'test':
            return data
        
        label = torch.tensor(self.label).unsqueeze(-1)
        
        return (data, label)


    @cached_property
    def label(self) -> int:
        classname = self.fingerprint['class_']
        try:
            classvalue = self.classlabel_mapping[classname]
        except KeyError:
            classvalue = self.classlabel_mapping[CLASSNAME_REMAP[classname]]
        return classvalue


    def __len__(self) -> int:
        return self.length



def load_volume_from_zarr(filepath: Path,
                          internal_path: str,
                          squeeze: bool = True) -> tuple[np.ndarray, dict]:
    """
    Eagerly load the volume and data fingerprint of the Zarr file at the given location.
    """
    data = zarr.convenience.open(filepath, mode='r')
    fingerprint = {k : v for k, v in data.attrs.items()}
    # select sub-dataset and load everything into main memory
    data = data[internal_path][...] if not squeeze else np.squeeze(data[internal_path][...])
    return (data, fingerprint)


class EagerSliceDatasetBuilder:
    """
    Programmatic dataset builder.
    Note that this builder loads the data from the zar files and *not* from
    the TIFF directories.
    """
    internal_path: str = 'downsampled/half'
    classlabel_mapping: dict[str, int] = DEFAULT_CLASSLABEL_MAPPING
    # TODO: factor hardcoded paths out -> bad!
    base_directory: Path = Path('/home/jannik/storage/wood/custom/')
    pretty_phase_name_map = {'val' : 'validation', 'train' : 'training', 'test' : 'testing'}

    @classmethod
    def build(cls,
              instances_ID: Iterable[str],
              phase: Literal['train'] | Literal['val'] | Literal['test'],
              axis: int,
              transform_configurations: Iterable[dict] | None = None
              ) -> list[EagerSliceDataset]:

        datasets = []
        if transform_configurations:
            transformer = Transformer(
                *from_configurations(transform_configurations)
            )
        else:
            transformer = None

        phase_name = cls.pretty_phase_name_map.get(phase, phase)
        desc = f'{phase_name} dataset build progress'
        wrapped_IDs = tqdm.tqdm(instances_ID, unit='dataset', desc=desc, leave=False)
        for ID in wrapped_IDs:
            wrapped_IDs.set_postfix_str(f'current_ID=\'{ID}\'')
            path = cls.get_path(ID)        
            volume, fingerprint = load_volume_from_zarr(path, cls.internal_path)
            dataset = EagerSliceDataset(
                phase=phase, volume=volume, fingerprint=fingerprint,
                transformer=transformer, classlabel_mapping=cls.classlabel_mapping,
                axis=axis
            )
            datasets.append(dataset)
        return datasets


    @classmethod
    def get_path(cls, ID: str) -> Path:
        for child in cls.base_directory.iterdir():
            if child.match(f'*/{ID}*'):
                return child
        raise FileNotFoundError(f'could not retrieve dataset with ID "{ID}" from '
                                f'basedir "{cls.base_directory}"')



class TiledEagerSliceDatasetBuilder:
    """
    Programmatic dataset builder for the tiled eager slice dataset.
    Note that this builder loads the data from the zar files and *not* from
    the TIFF directories.
    """
    internal_path: str = 'downsampled/half'
    classlabel_mapping: dict[str, int] = DEFAULT_CLASSLABEL_MAPPING
    # TODO: factor hardcoded paths out -> bad!
    base_directory: Path = Path('/home/jannik/storage/wood/custom/')
    pretty_phase_name_map = {'val' : 'validation', 'train' : 'training', 'test' : 'testing'}

    @classmethod
    def build(cls,
              instances_ID: Iterable[str],
              phase: Literal['train'] | Literal['val'] | Literal['test'],
              tileshape: tuple[int, int],
              z_spacing: int,
              axis: int,
              transform_configurations: Iterable[dict] | None = None
              ) -> list[EagerSliceDataset]:

        datasets = []
        if transform_configurations:
            transformer = Transformer(
                *from_configurations(transform_configurations)
            )
        else:
            transformer = None

        phase_name = cls.pretty_phase_name_map.get(phase, phase)
        desc = f'{phase_name} dataset build progress'
        wrapped_IDs = tqdm.tqdm(instances_ID, unit='dataset', desc=desc, leave=False)
        for ID in wrapped_IDs:
            wrapped_IDs.set_postfix_str(f'current_ID=\'{ID}\'')
            path = cls.get_path(ID)        
            volume, fingerprint = load_volume_from_zarr(path, cls.internal_path)
            dataset = TiledEagerSliceDataset(
                phase=phase, volume=volume, fingerprint=fingerprint,
                z_spacing=z_spacing, tileshape=tileshape,
                transformer=transformer, classlabel_mapping=cls.classlabel_mapping,
                axis=axis
            )
            datasets.append(dataset)
        return datasets



    @classmethod
    def get_path(cls, ID: str) -> Path:
        for child in cls.base_directory.iterdir():
            if child.match(f'*/{ID}*'):
                return child
        raise FileNotFoundError(f'could not retrieve datset with ID "{ID}" from '
                                f'basedir "{cls.base_directory}"')