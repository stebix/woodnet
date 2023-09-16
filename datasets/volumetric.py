"""
Implement 3D datasets for the wood CT data.

Jannik Stebani 2023
"""
import torch
import zarr

from collections.abc import Callable
from torch import Tensor
from pathlib import Path
from typing import Iterable

from custom.types import PathLike
from datasets import get_spatial_shape, DEFAULT_CLASSLABEL_MAPPING
from datasets.tiling import TileBuilder
from transformations import from_configurations
from transformations.transformer import Transformer

import tqdm.auto as tqdm


class TileDataset(torch.utils.data.Dataset):
    """
    Dataset for 3D tile based loading of the data.
    """
    eager: bool = True

    def __init__(self,
                 path: PathLike,
                 phase: str,
                 tileshape: tuple[int],
                 transformer: Callable | None = None,
                 classlabel_mapping: dict[str, int] | None = None,
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
    


class TileDatasetBuilder:
    """
    Build a 3D TileDataset programmatically.
    """
    internal_path: str = 'downsampled/half'
    classlabel_mapping: dict[str, int] = DEFAULT_CLASSLABEL_MAPPING
    # TODO: factor hardcoded paths out -> bad!
    base_directory = '/home/jannik/storage/wood/custom/'

    def build(cls, *IDs: str, phase: str, tileshape: tuple[int],
              transform_configurations: Iterable[dict] | None = None
              ) -> list[TileDataset]:
        
        datasets = []
        if transform_configurations:
            transformer = Transformer(
                *from_configurations(transform_configurations)
            )
        else:
            transformer = None

        wrapped_IDs = tqdm.tqdm(IDs, unit='dataset', desc='datasetbuilder')
        for ID in wrapped_IDs:
            wrapped_IDs.set_postfix({'current_ID' : str(ID)})
            path = cls.get_path(ID)        
            dataset = TileDataset(
                path=path, phase=phase, tileshape=tileshape,
                transformer=transformer,
                classlabel_mapping=cls.classlabel_mapping,
                internal_path=cls.internal_path
            )
            datasets.append(dataset)
        return datasets


    @classmethod
    def get_path(cls, ID: str) -> Path:
        base_directory = Path(cls.base_directory)
        for child in base_directory.iterdir():
            if child.match(f'*/{ID}*'):
                return child
        raise FileNotFoundError(f'could not retrieve datset with ID "{ID}" from '
                                f'basedir "{base_directory}"')

