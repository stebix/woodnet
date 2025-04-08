import logging

from pathlib import Path
from typing import Callable, Literal, TypeAlias
from functools import cached_property
from itertools import product

import torch.utils.data as torchdata
import numpy as np
import torch
import zarr

from torch import Tensor
import zarr.convenience

from woodnet.datasets.reader import Reader, deduce_reader_class
from woodnet.datasets.tiling import CylindricalVolumeTileBuilder, CuboidalVolumeTileBuilder
from woodnet.datasets.triaxial.typespecs import (
    Tilespec3D, TilespecND, Tileshape3D, Planespec3D
)
from woodnet.custom.types import PathLike

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)

ArrayLike: TypeAlias = np.ndarray


def get_spatial_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    assert len(shape) == 4, f'expected 4D shape, but got {shape}'
    return shape[-3:]


def generate_plane_slice(
    axis: int,
    index: int,
    tilespec: Tilespec3D
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

    Returns
    -------

    planespec : Planespec3D
        Specification of the plane as a tuple with the axis-wise
        index and tilespec slices that select the 2D plane.
    """
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


def generate_orthogonal_slices(
        tilespec: Tilespec3D,
        stride: tuple[int, int, int]
) -> tuple[Planespec3D, ...]:
    """
    Generate 3-tuples of slices that select the orthogonal planes
    inside the given tilespec.

    Parameters
    ----------

    tilespec : Tilespec3D
        Specification of the tile as 3-tuple of slices.
    
    stride : tuple[int, int, int]
        Stride along the axes to generate the planes.

    Returns
    -------

    slices : tuple[slice, ...]
        Tuple of 3-tuples of slices that select the orthogonal planes.
    """
    shape = get_shape_from(tilespec)
    slices = []
    for axis_index, (axis_size, stride) in enumerate(zip(shape, stride)):
        slices.append(
            [generate_plane_slice(axis_index, index, tilespec) for index in range(0, axis_size, stride)]
        )
    slices = tuple(planes for planes in product(*slices))
    return slices


def generate_maximal_tile(*args, **kwargs): raise NotImplementedError('implement this')


class LazyTriaxialDataset(torchdata.Dataset):

    def __init__(self,
                 path: PathLike,
                 internal_path: str,
                 phase: Literal['train', 'val'],
                 planestride: tuple[int, int, int],
                 tileshape: Tileshape3D | None = None,
                 reader_class: type[Reader] | None = None,
                 transformer: Callable | None = None,
                 classlabel_mapping: dict[str, int] | None = None,
                 tilegeneration_style: Literal['cylindrical', 'cuboidal'] = 'cylindrical',
                 fingerprint_path: Literal['root', 'internal'] = 'root',
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

        self.tilegeneration_style = tilegeneration_style
        self.tileshape, self.tiles = self._generate_tiles(tileshape)
        self.orthoplanes = self._generate_orthoplanes()


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
        
        (tileshape, tiles) : tuple of Tileshape3D and list[TileSpecND]
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
        return (tileshape, tiles)
    
    
    def _generate_orthoplanes(self) -> list[Planespec3D]:
        """
        Generate a single large list of all orthogonal planes from all
        the tiles in the dataset.
        """
        orthoplanes: list[Planespec3D] = []
        for tile in self.tiles:
            # TODO: this is ugly, make programmatically
            # we added the channel wildcard and made Tilespec3D -> TilespecND
            # so we need to remove the first slice such that orthoplanes can work
            tile3D = tile[1:] 
            orthoplanes.extend(
                generate_orthogonal_slices(tile3D, stride=self.planestride)
            )
        return orthoplanes
    
    
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
        orthoplane = self.orthoplanes[index]

        return orthoplane

        print(orthoplane)
        
        data = zarr.convenience.open(self.path, mode='r')[self.internal_path][*orthoplane]

        assert isinstance(data, np.ndarray)
        
        data = torch.tensor(data)

        if self.transformer:
            orthoplane = self.transformer(data)

        if self.phase == 'test':
            return data
        
        label = torch.tensor(self.label).unsqueeze_(-1)

        return (data, label)


    def __len__(self) -> int:
        """Number of elements in the dataset instance.
        In this case, this is the number of orthogonal planes.
        """
        return len(self.orthoplanes)
    
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
    
