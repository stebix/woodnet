"""
Implement 3D datasets for the wood CT data.

Jannik Stebani 2023
"""
import torch
import logging

from collections.abc import Callable
from functools import cached_property
from torch import Tensor
from pathlib import Path
from typing import Any, Iterable, Literal

import tqdm.auto as tqdm

from woodnet.custom.types import PathLike
from woodnet.datasets.tiling import VolumeTileBuilder
from woodnet.datasets.utils import get_spatial_shape
from woodnet.transformations import from_configurations
from woodnet.transformations.transformer import Transformer
from woodnet.datasets.reader import Reader, deduce_reader_class
from woodnet.datasets.setup import (InstanceFingerprint,
                                    DATA_CONFIGURATION, INSTANCE_MAPPING,
                                    INTERNAL_PATH, CLASSLABEL_MAPPING)


Tileshape3D = tuple[int, int, int]

DEFAULT_LOGGER_NAME = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

logger.info(f'Volumetric datasets are using internal path: {INTERNAL_PATH}')


class TileDataset(torch.utils.data.Dataset):
    """
    Dataset for 3D tile based loading of the data.

    Parameters
    ----------

    path : PathLike
        Path to the dataset on the file system.
    
    internal_path : str
        Path within the dataset file to the actual data.

    phase : Literal['train', 'val']
        Phase of the dataset: training or validation.

    tileshape : Tileshape3D
        Shape of the 3D tiles to extract from the volume data.
    
    reader_class : type[Reader] | None
        Reader class to use for loading the raw data from disk.
        The reader class will be deduced from the file suffix if `None`.
    
    transformer : Callable | None
        Transformation to apply to the data chunks before returning it.
    
    classlabel_mapping : dict[str, int] | None
        Mapping of class names to integer labels.
        Required for training phase datasets.

    Attributes
    ----------

    path : PathLike
        Path to the underlying dataset on the file system.
    
    internal_path : str
        Path within the dataset file to the actual data.

    reader : Reader
        Reader instance to load the data from disk. Must match the
        basal file format of the dataset.

    phase : Literal['train', 'val']
        Phase of the dataset: training or validation.

    tileshape : Tileshape3D
        Shape of the 3D tiles to extract from the entire volume data.

    transformer : Callable | None
        Transformation to apply to the data chunks before emitting
        them in the __getitem__ function.

    classlabel_mapping : dict[str, int] | None
        Mapping of class names to integer labels.
        Required for training phase datasets.
    
    volume : ndarray
        The entire volume data loaded from disk.
    
    fingerprint : dict
        Fingerprint metadata of the dataset instance.

    baseshape : tuple[int]
        The spatial shape of the volume data.

    tilebuilder : VolumeTileBuilder
        Tile builder instance to create the tile indices.

    label : int
        Integer label of the class deduced from the fingerprint.
        Constant for the entire dataset instance.
    """
    eager: bool = True

    def __init__(self,
                 path: PathLike,
                 internal_path: str,
                 phase: Literal['train', 'val'],
                 tileshape: Tileshape3D,
                 reader_class: type[Reader] | None = None,
                 transformer: Callable | None = None,
                 classlabel_mapping: dict[str, int] | None = None,
                 ) -> None:

        self.path = path
        self.internal_path = internal_path
        self.reader = self._init_reader(reader_class, path, internal_path)
        self.phase = phase
        self.tileshape = tileshape
        self.transformer = transformer
        self.classlabel_mapping = classlabel_mapping

        if self.phase == 'train' and classlabel_mapping is None:
            raise RuntimeError('Training phase dataset requires a '
                               'classlabel mapping!')
        
        # eager loading of data and fingerprint
        self.volume = self.reader.load_data()
        self.fingerprint = self.reader.load_fingerprint()
        # TODO: formalize this better
        self.baseshape = get_spatial_shape(self.volume.shape)
        radius = self.baseshape[-1] // 2
        self.tilebuilder = VolumeTileBuilder(
            baseshape=self.baseshape, tileshape=self.tileshape,
            radius=radius
        )
    
    @staticmethod
    def _init_reader(reader_class: type[Reader] | None, path: PathLike, internal_path: str) -> Reader:
        if reader_class is None:
            reader_class = deduce_reader_class(path)
        return reader_class(path=path, internal_path=internal_path)

    
    @cached_property
    def label(self) -> int:
        """Deduce the integer label of the class from the dataset fingerprint."""
        classname = self.fingerprint['class_']
        try:
            classvalue = self.classlabel_mapping[classname]
        except KeyError:
            raise KeyError(f'could not assign integer class value to class name \'{classname}\' - '
                           f'not found in classlabel mapping {self.classlabel_mapping.keys()}')
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
        """Number of elements in the dataset instance."""
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
    instance_mapping: dict[str, InstanceFingerprint] = INSTANCE_MAPPING
    classlabel_mapping: dict[str, int] = CLASSLABEL_MAPPING
    internal_path: str = INTERNAL_PATH
    pretty_phase_name_map = {'val' : 'validation', 'train' : 'training', 'test' : 'testing'}

    def build(cls,
              dataset_class: type,
              instances_ID: Iterable[str],
              phase: Literal['train', 'val', 'test'],
              tileshape: Tileshape3D,
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
        try:
            fingerprint = cls.instance_mapping[ID]
        except KeyError:
            raise FileNotFoundError(f'could not retrieve dataset instance with ID "{ID}" - '
                                    f'check if ID is present in the data configuration!')

        return fingerprint.location



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
    
