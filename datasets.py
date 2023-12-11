import torch
import numpy as np
import zarr

from typing import Literal, Optional, Union

from tiling import TileBuilder
from dataobjects import AbstractSlice, Volume
from custom.types import PathLike


DEFAULT_CLASSLABEL_MAPPING = {
    'ahorn' : 0,
    'kiefer' :  1
}

Tensor = torch.Tensor


def add_channel_dim(array: np.ndarray) -> np.ndarray:
    """Add fake channel dimension."""
    return array[np.newaxis, ...]


def get_spatial_shape(shape: tuple[int]) -> tuple[int]:
    """Get spatial shape for 4D inputs"""
    return shape[1:]


class SliceDataset(torch.utils.data.Dataset):
    """
    Dataset consisting of many 2D slices (H x W)
    """
    def __init__(self,
                 phase: Literal['train', 'test'],
                 slices: list[AbstractSlice],
                 classlabel_mapping: Optional[dict[str, int]] = None,
                 transformer: Optional[callable] = None,
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
                 volume: Volume,
                 transformer: Optional[callable] = None,
                 classlabel_mapping: Optional[dict[str, int]] = None,
                 axis: int = 0,
                 ) -> None:

        self.phase = phase
        self.axis = axis
        self.fingerpint = volume.fingerprint
        self.volume = np.swapaxes(volume.data, 0, axis)
        self.transformer = transformer

        if self.phase == 'train' and classlabel_mapping is None:
            raise RuntimeError('Training phase dataset requires a '
                               'classlabel mapping!')

        self.classlabel_mapping = classlabel_mapping
        # we assume that for the loaded volume the class is
        #  constant 
        self._label = torch.tensor(
            self.classlabel_mapping[self.fingerpint.class_]
        ).unsqueeze_(-1)


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | Tensor:        
        data = torch.tensor(add_channel_dim(self.volume[index]))

        if self.transformer:
            data = self.transformer(data)

        if self.phase == 'test':
            return data
        
        return (data, self._label)
    

    def __len__(self) -> int:
        return self.volume.shape[0]



class TileDataset(torch.utils.data.Dataset):
    """
    Dataset for 3D tile based loading of the data.
    """
    eager: bool = True

    def __init__(self,
                 path: PathLike,
                 phase: str,
                 tileshape: tuple[int],
                 transformer: Optional[callable] = None,
                 classlabel_mapping: Optional[dict[str, int]] = None,
                 internal_path: str = 'downsampled/half') -> None:

        self.path = path
        self.phase = phase
        self.tileshape = tileshape
        self.transformer = transformer
        self.classlabel_mapping = classlabel_mapping
        self.internal_path = internal_path

        if self.phase == 'train' and classlabel_mapping is None:
            raise RuntimeError('Training phase dataset requires a '
                               'classlabel mapping!')
        # underlying zarr storage with metadata fingerprint
        self.data = zarr.convenience.open(self.path, mode='r')
        self.fingerprint = {k : v for k, v in self.data.attrs.items()}
        # lazily loaded volume data
        self.volume = self.data[self.internal_path][...]

        self.baseshape = get_spatial_shape(self.volume.shape)
        # TODO: formalize this better
        radius = self.baseshape[-1] // 2
        self.tilebuilder = TileBuilder(
            baseshape=self.baseshape, tileshape=self.tileshape,
            radius=radius
        )
    

    def _load_volume(self):
        if self.eager:
            return self.data[self.internal_path][...]
        else:
            return self.data[self.internal_path]

    
    @property
    def label(self) -> int:
        return self.classlabel_mapping[self.fingerprint['class_']]


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

        if self.phase == 'test':
            return subvolume
        
        label = torch.tensor(self.label).unsqueeze_(-1)

        return (subvolume, label)


    def __len__(self) -> int:
        return len(self.tilebuilder.tiles)
    

    def __str__(self) -> str:
        has_transformer = True if self.transformer else False
        s = f'{self.__class__.__name__}('
        infos = ', '.join((
            f"path='{self.path}'", f"phase='{self.phase}'",
            f"baseshape={self.baseshape}", f"tileshape={self.tileshape}",
            f"classlabel_mapping={self.classlabel_mapping}",
            f"has_transformer={has_transformer}"
        ))
        return ''.join((s, infos, ')'))
    

    def __repr__(self) -> str:
        return str(self)