"""
Implements the triaxial dataset that utilizes the three orthogonal X/Y/Z-planes
inside the specified volume.

@jsteb 2024
"""
import logging
import numpy as np
import torch
import torch.utils.data as torchdata
import zarr
import tqdm.auto as tqdm

from pathlib import Path
from functools import partial, cached_property
from itertools import product
from collections.abc import Callable, Sequence, Iterable
from typing import Literal
from torch import Tensor

from woodnet.datasets.constants import CLASSNAME_REMAP
from woodnet.datasets.tiling import TileBuilder
from woodnet.custom.types import PathLike
from woodnet.inference.parametrized_transforms import ParametrizedTransform


LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


ArrayLike = np.ndarray | Tensor
TileShape = tuple[int, int, int]
TileSlice = tuple[slice, ...]


def generate_plane_slice(axis: int, index: int) -> tuple[slice]:
    """
    Generate the index & slice tuple that selects the plane along the
    given axis and position.
    """
    slices = [slice(None, None), slice(None, None), slice(None, None)]
    slices[axis] = index
    return tuple(slices)


def generate_orthogonal_slices(shape: tuple[int, int, int],
                               stride: tuple[int, int, int]) -> tuple[slice]:
    """
    Generate 3-tuples of slices that select the orthogonal planes.
    """
    slices = []
    for axis_index, (axis_size, stride) in enumerate(zip(shape, stride)):
        slices.append(
            [generate_plane_slice(axis_index, index) for index in range(0, axis_size, stride)]
        )
    slices = tuple(planes for planes in product(*slices))
    return slices


def select_stack_func(backend: Literal['numpy', 'torch']) -> Callable[[Sequence[ArrayLike]], ArrayLike]:
    """
    Select the `stack` function for the backends `numpy` and `torch` and provide it as
    a single-argument function.
    """
    if backend == 'numpy':
        return partial(np.stack, axis=0)
    elif backend == 'torch':
        return partial(torch.stack, dim=0)
    else:
        raise ValueError(f'invalid backend choice \'{backend}\': must be \'numpy\' or \'torch\'')
    

def generate_orthogonal_planes(volume: ArrayLike,
                               stride: tuple[int, int, int]
                               ) -> tuple[ArrayLike]:
    """
    Generate the orthogonal planes along the three canonical dimensions.
    Assumes that the three trailing dimensions are the spatial dimensions.
    """
    backend = 'numpy' if isinstance(volume, np.ndarray) else 'torch'
    stack_fn = select_stack_func(backend)
    orthoplanes = []
    shape = volume.shape
    tri_plane_specs = generate_orthogonal_slices(shape, stride)
    for tri_plane_spec in tri_plane_specs:
        planes = []
        for plane_spec in tri_plane_spec:
            planes.append(volume[plane_spec])
        orthoplanes.append(stack_fn(planes))
    return tuple(orthoplanes)


def is_square(shape: Iterable[int],
              dims: tuple[int] | None = None) -> bool:
    """
    Check if shape-like object is square along the indicate dimensions.
    Defaults to None, meaning that all dimensons are considered.    
    """
    if dims is None:
        dims = np.s_[:]
    else:
        dims = np.array(dims)
    sizes = np.array(shape)[dims]
    a = sizes[0]
    if all(a == s for s in sizes[1:]):
        return True
    return False


def generate_maximal_tile_OLD(shape: tuple[int, int, int],
                          prepend_wildcards: int = 1) -> tuple[slice]:
    """
    Generate the slices for the single maximal tile that encompasses
    the largest suqare fitting inside the cylindrical-circular data region.
    Along the vertical z-axis, the full volume is utilized.
    
    Parameters
    ==========
    
    shape : tuple[int, int, int]
        Base 3D volume shape.
        
    prepend_wildcards : int
        Number of wildcard, i.e. full-selecting slice
        objects prepended before the spatial slices.
        Defaults to 1.
    """
    if not is_square(shape, dims=(1, 2)):
        raise ValueError(f'cannot generate maximal tile for non-square '
                         f'shape in axes (1,2): {shape}')
    radius = shape[0] // 2
    edge = np.sqrt(2) * radius
    # intial starting positions
    axis_1_start = shape[1] // 2 - np.rint(edge / 2).astype(int)
    axis_2_start = shape[2] // 2 - np.rint(edge / 2).astype(int)
    # final stopping positions
    axis_1_stop = np.rint(axis_1_start + edge).astype(int)
    axis_2_stop = np.rint(axis_2_start + edge).astype(int)
    # along 0-th 'vertical' axis everything is selected
    axis_0_wildcard = slice(None, None)
    axis_1_slice = slice(axis_1_start, axis_1_stop)
    axis_2_slice = slice(axis_2_start, axis_2_stop)
    wildcards = tuple(slice(None, None) for _ in range(prepend_wildcards))
    return tuple((*wildcards, axis_0_wildcard, axis_1_slice, axis_2_slice))



def generate_maximal_tile(shape: tuple[int, int, int],
                          prepend_wildcards: int = 1) -> tuple[slice]:
    """
    Generate the slices for the single maximal tile that encompasses
    the largest suqare fitting inside the cylindrical-circular data region.
    Along the vertical z-axis, the full volume is utilized.
    
    Parameters
    ==========
    
    shape : tuple[int, int, int]
        Base 3D volume shape.
        
    prepend_wildcards : int
        Number of wildcard, i.e. full-selecting slice
        objects prepended before the spatial slices.
        Defaults to 1.
    """
    if not is_square(shape, dims=(1, 2)):
        raise ValueError(f'cannot generate maximal tile for non-square '
                         f'shape in axes (1,2): {shape}')
    radius = shape[0] // 2
    edge = np.sqrt(2) * radius
    # intial starting positions
    start = shape[1] // 2 - np.rint(edge / 2).astype(int)
    # final stopping positions
    stop = np.rint(start + edge).astype(int)
    # along 0-th 'vertical' axis the range to make the tile a cube is selected
    axis_0_slice = slice(0, stop-start)
    axis_slice = slice(start, stop)
    wildcards = tuple(slice(None, None) for _ in range(prepend_wildcards))
    return tuple((*wildcards, axis_0_slice, axis_slice, axis_slice))


def size_from_slice(slc: slice) -> int:
    return slc.stop - slc.start

def get_spatial_shape(shape: tuple[int]) -> tuple[int]:
    """Get spatial shape for 4D inputs"""
    return shape[1:]


class TriaxialDataset(torchdata.Dataset):
    """
    Triaxial dataset with orthogonal images concatenated along the
    channel dimension.

    The dataset can provide the orthogonal images for the maximal available
    volume and many small subvolumes.
    """
    def __init__(self,
                 path: PathLike,
                 phase: Literal['train', 'val'],
                 planestride: tuple[int, int, int],
                 tileshape: TileShape | None = None,
                 transformer: Callable | None = None,
                 classlabel_mapping: dict[str, int] | None = None,
                 internal_path: str = 'downsampled/half'
                 ) -> None:

        super().__init__()

        self.path = Path(path)
        self.phase = phase
        self.planestride = planestride
        self.transformer = transformer
        self.classlabel_mapping = classlabel_mapping
        self.internal_path = internal_path
        
        if self.phase in {'train', 'val'} and classlabel_mapping is None:
            raise RuntimeError(f'Phase \'{self.phase}\' dataset requires a '
                               f'classlabel mapping!')
        # underlying zarr storage with metadata fingerprint
        self.data = zarr.convenience.open(self.path, mode='r')
        self.fingerprint = {k : v for k, v in self.data.attrs.items()}
        # lazily loaded volume data
        self.volume = self.data[self.internal_path][...]
        self.baseshape = get_spatial_shape(self.volume.shape)
        self.tileshape, self.tiles = self._generate_tiles(tileshape)
        self.orthoplanes = self._generate_orthoplanes()
        
    def _generate_tiles(self,
                        tileshape: TileShape | None
                        ) -> tuple[TileShape, list[TileSlice]]:
        """Generate the tiles as 3-tuples of slice objects.
        Selects the maximally available tile or the tile shape builder depending
        on input.
        
        Parameters
        ==========
        
        tileshape: tuple[int, int, int] or None
            Desired tileshape. For `None`, the maximallly available
            tile volume is selected.
            
        Returns
        =======
        
        (tileshape, tiles) : tuple of TileShape and list[TileSlice]
            The actual tileshape and the slices that select the tiles
            from the full volume.
        """
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
            builder = TileBuilder(
                baseshape=self.baseshape, tileshape=tileshape,
                radius=radius
            )
            tiles = builder.tiles
        return (tileshape, tiles)
    
    
    def _generate_orthoplanes(self) -> list[ArrayLike]:
        orthoplanes = []
        for tile in self.tiles:
            subvolume = np.squeeze(self.volume[tile])
            orthoplanes.extend(
                generate_orthogonal_planes(volume=subvolume, stride=self.planestride)
            )
        return orthoplanes
    
    
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
        orthoplane = self.orthoplanes[index]
        orthoplane = torch.tensor(orthoplane)

        if self.transformer:
            orthoplane = self.transformer(orthoplane)

        if self.phase == 'test':
            return orthoplane
        
        label = torch.tensor(self.label).unsqueeze_(-1)

        return (orthoplane, label)


    def __len__(self) -> int:
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


from woodnet.datasets.constants import DEFAULT_CLASSLABEL_MAPPING
from woodnet.transformations.transformer import Transformer
from woodnet.transformations.buildtools import from_configurations




class TriaxialDatasetBuilder:
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
              instances_ID: Iterable[str],
              phase: Literal['train', 'val', 'test'],
              tileshape: TileShape,
              planestride: tuple[int, int, int],
              transform_configurations: Iterable[dict] | None = None,
              **kwargs
              ) -> list[TriaxialDataset]:
        
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
            dataset = TriaxialDataset(
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
        for child in cls.base_directory.iterdir():
            if child.match(f'*/{ID}*'):
                return child
        raise FileNotFoundError(f'could not retrieve datset with ID "{ID}" from '
                                f'basedir "{cls.base_directory}"')
