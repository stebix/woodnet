import logging

from pathlib import Path
from collections.abc import Iterable, Callable
from typing import Literal, TypeAlias
from functools import cached_property
from itertools import product

import torch.utils.data as torchdata
import numpy as np
import torch
import zarr
import tqdm

from torch import Tensor
import zarr.convenience

from woodnet.datasets.reader import Reader, deduce_reader_class
from woodnet.datasets.tiling import CylindricalVolumeTileBuilder, CuboidalVolumeTileBuilder
from woodnet.datasets.tiling.utils import is_square

from woodnet.datasets.setup import (InstanceFingerprint,
                                    INTERNAL_PATH, CLASSLABEL_MAPPING, INSTANCE_MAPPING)
from woodnet.custom.types import PathLike
from woodnet.transformations.transformer import Transformer
from woodnet.transformations.buildtools import from_configurations
from woodnet.datasets.triaxial.typespecs import(
    Tilespec3D, TilespecND, Tileshape3D, Planespec3D, TriaxPlanesSpec, TilePlanespec3D
)

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


ArrayLike: TypeAlias = np.ndarray



def get_spatial_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    assert len(shape) == 4, f'expected 4D shape, but got {shape}'
    return shape[-3:]


def generate_plane_specification(
    axis: int,
    index: int,
    tilespec: Tilespec3D | None
) -> Planespec3D:
    """
    Generate the index & slice tuple that selects the 2D plane along the
    given axis and position within the given tile (provided as `tilespec`).

    Parameters
    ----------

    axis : int
        Axis index to select the plane from.

    index : int
        Position index along the axis.

    tilespec : Tilespec3D | None
        Restrict the plane slice to the given tilespec, if provided.
        Otherwise, the full axis range is used.

    Returns
    -------

    planespec : Planespec3D
        Specification of the plane as a tuple with the axis-wise
        index and tilespec slices that select the 2D plane.
    """
    tilespec = tilespec if tilespec is not None else (slice(None), slice(None), slice(None))
    if axis == 0:
        return tuple((index, tilespec[1], tilespec[2]))
    elif axis == 1:
        return tuple((tilespec[0], index, tilespec[2]))
    elif axis == 2:
        return tuple((tilespec[0], tilespec[1], index))
    else:
        raise ValueError(f'invalid axis: {axis}')


def get_shape_from(tilespec: Tilespec3D) -> tuple[int, int, int]:
    """
    Get the shape of the 3D volume from the tilespec.

    Parameters
    ----------

    tilespec : Tilespec3D
        3-tuple of slice objects that select the 3D volume.

    Returns
    -------

    shape : tuple[int, int, int]
        Shape of the 3D volume.
    """
    return tuple((s.stop - s.start) for s in tilespec)


def generate_triaxial_planes_specifications(
    tilespec: Tilespec3D,
    stride: tuple[int, int, int],
    *,
    mode: Literal['relative', 'absolute'] = 'relative'
) -> tuple[TriaxPlanesSpec, ...]:
    """
    Generate all planespecs for the given tilespec with the given stride.
    """
    shape = get_shape_from(tilespec)

    if mode == 'relative':
        tilespec_mode = None
        index_offset = (0, 0, 0)
    elif mode == 'absolute':
        raise NotImplementedError('implement this')
        tilespec_mode = tilespec
        index_offset = (tilespec[0].start, tilespec[1].start, tilespec[2].start)
    else:
        raise ValueError(f'Invalid mode: {mode} - only \'relative\' and \'absolute\' are supported')
    
    planes = []
    for axis_index, (axis_size, stride, offset) in enumerate(zip(shape, stride, index_offset)):
        axis_planes = [
            generate_plane_specification(axis_index, index+offset, tilespec_mode) for index in range(0, axis_size+offset, stride)
        ]
        planes.append(tuple(axis_planes))

    triaxial_planes_spec = tuple(TriaxPlanesSpec(*planes) for planes in product(*planes))
    return triaxial_planes_spec


def generate_layered_tile_triaxplanes_specifications(
        tilespecs: list[Tilespec3D],
        stride: tuple[int, int, int]
) -> tuple[TilePlanespec3D, ...]:
    """
    Layered tile plane specs: generate tuple of tile indices TriaxPlanesSpec
    """
    tile_triaxplanes_spec = []

    for tileindex, tilespec in enumerate(tilespecs):
        triaxial_planes = generate_triaxial_planes_specifications(
            tilespec=tilespec,
            stride=stride
        )
        tile_triaxial_planes = tuple(
            (tileindex, elem) for elem in triaxial_planes
        )
        tile_triaxplanes_spec.extend(tile_triaxial_planes)

    return tuple(tile_triaxplanes_spec)


def generate_maximal_tile(*args, **kwargs): raise NotImplementedError('implement this')


class LazyCachingTriaxialDataset(torchdata.Dataset):

    def __init__(self,
                 path: PathLike,
                 internal_path: str,
                 phase: Literal['train', 'val', 'test'],
                 planestride: tuple[int, int, int],
                 tileshape: Tileshape3D | None = None,
                 max_tile_count: int | None = None,
                 reader_class: type[Reader] | None = None,
                 transformer: Callable | None = None,
                 classlabel_mapping: dict[str, int] | None = None,
                 tilegeneration_style: Literal['cylindrical', 'cuboidal'] = 'cylindrical',
                 fingerprint_path: Literal['root', 'internal'] = 'root',
                 populate_cache: bool = False
                 ) -> None:

        super().__init__()

        self.path = Path(path)
        self.phase = phase
        self.planestride = planestride
        self.transformer = transformer
        self.classlabel_mapping = classlabel_mapping
        self.internal_path = internal_path
        self.reader = self._init_reader(reader_class, path, internal_path, fingerprint_path)
        
        if self.phase in {'train', 'val'} and classlabel_mapping is None:
            raise RuntimeError(f'Phase \'{self.phase}\' dataset requires a '
                               f'classlabel mapping!')
        
        # load from underlying storage with metadata fingerprint
        # self.volume = self.reader.load_data()
        self.fingerprint = self.reader.load_fingerprint()

        # TODO: fix with lazy loading
        self.baseshape = get_spatial_shape(
            zarr.convenience.open(self.path, mode='r')[self.internal_path].shape
        )

        self.max_tile_count = max_tile_count
        self.tilegeneration_style = tilegeneration_style
        self.tileshape, self.tiles = self._generate_tiles(tileshape)

        self.item_specs = self._generate_itemspecs()

        self._tile_volume_cache: dict[int, np.ndarray] = {}

        if populate_cache:
            volume = self.reader.load_data()
            for tileindex, tile in enumerate(self.tiles):
                print(f'Populating cache for tile {tileindex} / {len(self.tiles)}')
                tilevolume = np.squeeze(volume[tile])
                self._tile_volume_cache[tileindex] = tilevolume


    @staticmethod
    def _init_reader(
        reader_class: type[Reader] | None,
        path: PathLike,
        internal_path: str,
        fingerprint_path: Literal['root', 'internal'] = 'root'
    ) -> Reader:
        if reader_class is None:
            reader_class = deduce_reader_class(path)
        
        if fingerprint_path == 'root':
            fingerprint_path = '/'
        elif fingerprint_path == 'internal':
            fingerprint_path = internal_path
        else:
            raise ValueError(f'Invalid fingerprint path \'{fingerprint_path}\' - '
                             f'only \'root\' and \'internal\' are supported')
        
        return reader_class(path=path, internal_path=internal_path, fingerprint_path=fingerprint_path)
    
        
    def _generate_tiles(self,
                        tileshape: tuple[int, int, int] | None
                        ) -> tuple[Tileshape3D, tuple[TilespecND, ...]]:
        """Generate the tiles as 3-tuples of slice objects.
        Selects the maximally available tile or the tile shape builder depending
        on input.
        
        Parameters
        ----------
        
        tileshape: tuple[int, int, int] or None
            Desired tileshape. For `None`, the maximallly available
            tile volume is selected.
            
        Returns
        -------
        
        (tileshape, tiles) : tuple of TileShape and list[TileSlice]
            The actual tileshape and the slices that select the tiles
            from the full volume.
        """

        if self.tilegeneration_style == 'cylindrical':
            if tileshape is None:
                # generate maximally available tile with layout (0 : channel, 1 : axis0, 2 : axis1, 3 : axis2)
                tile = generate_maximal_tile(self.baseshape, prepend_wildcards=1)
                tiles = [tile]
                tileshape = (
                    self.baseshape[0],
                    tile[2].stop - tile[2].start,
                    tile[3].stop - tile[3].start
                )
            else:
                radius = self.baseshape[-1] // 2
                builder = CylindricalVolumeTileBuilder(
                    baseshape=self.baseshape, tileshape=tileshape,
                    radius=radius
                )
                tiles = builder.tiles
        elif self.tilegeneration_style == 'cuboidal':
            builder = CuboidalVolumeTileBuilder(
                baseshape=self.baseshape, tileshape=tileshape
            )
            tiles = builder.tiles
        else:
            raise ValueError(f'invalid tile generation style \'{self.tilegeneration_style}\' - '
                             f'only \'cylindrical\' and \'cuboidal\' are supported')
        
        logger.info(
            f'Generated {len(tiles)} tiles with shape {tileshape} from '
            f'{self.baseshape} with data source \'{self.path}\''
        )
        if self.max_tile_count is not None:
            if len(tiles) > self.max_tile_count:
                logger.info(f'Number of tiles ({len(tiles)}) exceeds max tile count ({self.max_tile_count}) - '
                            f'truncating to {self.max_tile_count} tiles')
                tiles = tiles[:self.max_tile_count]

        return (tileshape, tiles)
    
    
    def _generate_itemspecs(self) -> list[TilePlanespec3D]:
        """
        Generate a single large list of all orthogonal planes from all
        the tiles in the dataset.
        """

        # TODO: this is ugly, make programmatically
        # we added the channel wildcard and made Tilespec3D -> TilespecND
        # so we need to remove the first slice such that orthoplanes can work
        tiles = [tile[1:] for tile in self.tiles]
        itemspecs = generate_layered_tile_triaxplanes_specifications(
            tilespecs=tiles,
            stride=self.planestride
        )
        return itemspecs


    def _get_tile(self, tileindex: int) -> np.ndarray:
        """
        Retrieve the tile volume from the underlying storage.
        """
        if tileindex in self._tile_volume_cache:
            return self._tile_volume_cache[tileindex]
        
        logger.debug(f'Cache miss - loading tile volume {tileindex} / {len(self.tiles)} from {self.path}')
        tile = self.tiles[tileindex]

        import time
        start = time.time()
        tilevolume = zarr.convenience.open(self.path, mode='r')[self.internal_path][*tile]
        end = time.time()
        logger.debug(f'Loading tile took {end - start:.3f}s')

        # TODO: improve this: for triaxial channel is along the orthoplanes
        tilevolume = np.squeeze(tilevolume)
        assert tilevolume.ndim == 3, f'expected 3D volume, but got {tilevolume.ndim}D volume'

        self._tile_volume_cache[tileindex] = tilevolume
        return tilevolume

    
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
    
    @staticmethod
    def _collate_planes(
        volume: np.ndarray,
        triax_planes_spec: TriaxPlanesSpec
    ) -> np.ndarray:
        """
        Collate the planes of the volume according to the triax_planes_spec.
        """
        zplane, yplane, xplane = triax_planes_spec
        return np.stack((volume[zplane], volume[yplane], volume[xplane]), axis=0)


    def __getitem__(self, index: int) -> tuple[Tensor] | Tensor:
        """
        Retrieve dataset item: tuple of tensor for training phase
        (data and label) or test phase (single tensor).
        """
        tileindex, triax_planes_spec = self.item_specs[index]

        tilevolume = self._get_tile(tileindex)
        data = self._collate_planes(tilevolume, triax_planes_spec)

        # TODO: remove later

        # write an assertion that the last 3 elements of self.tileshape are equal integer values
        assert is_square(self.tileshape), 'tileshape must be cube, otherwise check below is wrong'
        assert data.shape == (3, self.tileshape[-1], self.tileshape[-2]), f'expected shape {(3, self.tileshape[-1], self.tileshape[-2])} but got {data.shape}'
        assert isinstance(data, np.ndarray)
        
        data = torch.tensor(data)

        if self.transformer:
            data = self.transformer(data)

        if self.phase == 'test':
            return data
        
        label = torch.tensor(self.label).unsqueeze_(-1)

        return (data, label)


    def __len__(self) -> int:
        """Number of elements in the dataset instance.
        In this case, this is the number of orthogonal planes.
        """
        return len(self.item_specs)
    
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
    



# BWAHAHAHAHHAHA
class LazyCachingTriaxialDatasetBuilder:
    """
    Build a 3D TileDataset programmatically.

    Thin class, basically acts as a namespace. Maybe move to module?

    Attributes
    ----------

    instance_mapping : dict[str, InstanceFingerprint]
        Mapping of instance IDs to instance fingerprints.
        Utilized to retrieve the instance data via the set unqiue ID string.
        The fingerprints must at least provide the location of
        the instance data on disk.

    internal_path : str
        Internal path to the data inside the storage container.

    classlabel_mapping : dict[str, int]
        Mapping of class names to integer labels.

    pretty_phase_name_map : dict[str, str]
        Mapping of phase names to pretty-printable names.
    """
    instance_mapping: dict[str, InstanceFingerprint] = INSTANCE_MAPPING
    internal_path: str = INTERNAL_PATH
    classlabel_mapping: dict[str, int] = CLASSLABEL_MAPPING
    pretty_phase_name_map = {'val' : 'validation', 'train' : 'training', 'test' : 'testing'}

    def build(cls,
              instances_ID: Iterable[str],
              phase: Literal['train', 'val', 'test'],
              tileshape: Tileshape3D,
              planestride: tuple[int, int, int],
              transform_configurations: Iterable[dict] | None = None,
              **kwargs
              ) -> list[LazyCachingTriaxialDataset]:
        """
        Build the TriaxialDataset instances from the provided instance IDs.

        Parameters
        ----------

        instances_ID : Iterable[str]
            Iterable of instance IDs to build the datasets from. The IDs
            must be present in the instance mapping.

        phase : Literal['train', 'val', 'test']
            Phase of the dataset: training, validation or testing.

        tileshape : TileShape
            Desired tileshape for the datasets.

        planestride : tuple[int, int, int]
            Stride along the three axes to generate the orthogonal planes.
            Lower values generate more orthoplane elements.

        transform_configurations : Iterable[dict] | None
            Iterable of transformation configurations to apply to the
            orthoplane elements. Defaults to None, i.e. no transformations.

        Returns
        -------

        datasets : list[TriaxialDataset]
            List of TriaxialDataset instances.
        """
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
            dataset = LazyCachingTriaxialDataset(
                path=path, phase=phase,
                planestride=planestride,
                tileshape=tileshape,
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