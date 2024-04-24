"""
Implement 3D datasets for the wood CT data.

Jannik Stebani 2023
"""
import torch
import zarr

from collections.abc import Callable
from functools import cached_property
from torch import Tensor
from pathlib import Path
from typing import Iterable, Literal


import tqdm.auto as tqdm

from woodnet.custom.types import PathLike
from woodnet.datasets.constants import CLASSNAME_REMAP, DEFAULT_CLASSLABEL_MAPPING
from woodnet.datasets.tiling import TileBuilder
from woodnet.datasets.utils import get_spatial_shape
from woodnet.transformations import from_configurations
from woodnet.transformations.transformer import Transformer



class TileDataset(torch.utils.data.Dataset):
    """
    Dataset for 3D tile based loading of the data.
    """
    eager: bool = True

    def __init__(self,
                 path: PathLike,
                 phase: Literal['train', 'val'],
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

    
    @cached_property
    def label(self) -> int:
        classname = self.fingerprint['class_']
        try:
            classvalue = self.classlabel_mapping[classname]
        except KeyError:
            classvalue = self.classlabel_mapping[CLASSNAME_REMAP[classname]]
        return classvalue


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

    
    def _make_info_str(self) -> str:
        has_transformer = True if self.transformer else False
        infos = ', '.join((
            f"path='{self.path}'", f"phase='{self.phase}'",
            f"baseshape={self.baseshape}", f"tileshape={self.tileshape}",
            f"classlabel_mapping={self.classlabel_mapping}",
            f"has_transformer={has_transformer}"
        ))
        return infos
    

    def __str__(self) -> str:
        s = f'{self.__class__.__name__}('
        infos = self._make_info_str()
        return ''.join((s, infos, ')'))
    

    def __repr__(self) -> str:
        return str(self)
    


class BaseTileDatasetBuilder:
    """
    Build a 3D TileDataset programmatically.

    Thin class, basically acts as a namespace. Maybe move to module?
    """
    internal_path: str = 'downsampled/half'
    classlabel_mapping: dict[str, int] = DEFAULT_CLASSLABEL_MAPPING
    # TODO: factor hardcoded paths out -> bad!
    base_directory: Path = Path('/home/jannik/storage/wood/custom/')
    pretty_phase_name_map = {'val' : 'validation', 'train' : 'training', 'test' : 'testing'}

    def build(cls,
              dataset_class: type,
              instances_ID: Iterable[str],
              phase: Literal['train', 'val', 'test'],
              tileshape: tuple[int, int, int],
              transform_configurations: Iterable[dict] | None = None,
              **kwargs
              ) -> list[TileDataset]:
        
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
            dataset = dataset_class(
                path=path, phase=phase, tileshape=tileshape,
                transformer=transformer,
                classlabel_mapping=cls.classlabel_mapping,
                internal_path=cls.internal_path,
                **kwargs
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



class TileDatasetBuilder(BaseTileDatasetBuilder):
    """
    Builder for the standard TileDataset for training, validation and testing.
    """
    def build(cls,
              instances_ID: Iterable[str],
              phase: Literal['train'] | Literal['val'] | Literal['test'],
              tileshape: tuple[int, int, int],
              transform_configurations: Iterable[dict] | None = None
              ) -> list[TileDataset]:
        
        return super().build(TileDataset, instances_ID, phase,
                             tileshape, transform_configurations)
    
