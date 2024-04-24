"""
Implement 2D datasets for the wood CT data.

Jannik Stebani 2023
"""
import torch
import numpy as np

from collections.abc import Callable, Iterable, Mapping
from functools import cached_property
from torch import Tensor
from typing import Optional, Literal, Union
from pathlib import Path

import tqdm.auto as tqdm

from woodnet.datasets.utils import add_channel_dim
from woodnet.dataobjects import AbstractSlice, Volume
from woodnet.datasets.constants import DEFAULT_CLASSLABEL_MAPPING, CLASSNAME_REMAP
from woodnet.inference.parametrized_transforms import ParametrizedTransform
from woodnet.transformations.transformer import Transformer
from woodnet.transformations.buildtools import from_configurations 

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


import zarr

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
        raise FileNotFoundError(f'could not retrieve datset with ID "{ID}" from '
                                f'basedir "{cls.base_directory}"')
