"""
Implements the triaxial dataset that utilizes the three orthogonal X/Y/Z-planes
inside the specified volume.

@jsteb 2024
"""
import logging
import numpy as np
import torch
import torch.utils.data as torchdata
import tqdm.auto as tqdm

from pathlib import Path
from functools import partial, cached_property
from itertools import product
from collections.abc import Callable, Sequence, Iterable
from typing import Literal
from torch import Tensor

from woodnet.datasets.setup import (InstanceFingerprint,
                                    INTERNAL_PATH, CLASSLABEL_MAPPING, INSTANCE_MAPPING)
from woodnet.datasets.tiling import VolumeTileBuilder
from woodnet.custom.types import PathLike
from woodnet.transformations.transformer import Transformer
from woodnet.transformations.buildtools import from_configurations
from woodnet.datasets.reader import Reader, deduce_reader_class

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


ArrayLike = np.ndarray | Tensor
TileShape = tuple[int, int, int]
TileSlice = tuple[slice, ...]


def generate_plane_slice(axis: int, index: int) -> tuple[slice]:
    """
    Generate the index & slice tuple that selects the 2D plane along the
    given axis and position.

    Parameters
    ----------

    axis : int
        Axis index to select the plane from.

    index : int
        Position index along the axis.

    Returns
    -------

    slice : tuple[slice]
        3-tuple of slice objects that select the 2D plane.
    """
    slices = [slice(None, None), slice(None, None), slice(None, None)]
    slices[axis] = index
    return tuple(slices)


def generate_orthogonal_slices(shape: tuple[int, int, int],
                               stride: tuple[int, int, int]) -> tuple[slice, ...]:
    """
    Generate 3-tuples of slices that select the orthogonal planes.

    Parameters
    ----------

    shape : tuple[int, int, int]
        Shape of the 3D volume from which to generate the planes.
    
    stride : tuple[int, int, int]
        Stride along the axes to generate the planes.

    Returns
    -------

    slices : tuple[slice, ...]
        Tuple of 3-tuples of slices that select the orthogonal planes.
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
    a single-argument function by fixing the axis or dim argument to the first axis.

    Parameters
    ----------

    backend : Literal['numpy', 'torch']
        Backend choice for the stack function.
    
    Returns
    -------

    stack_fn : Callable[[Sequence[ArrayLike]], ArrayLike]
        Function that stacks the input sequence of arrays along the first axis.

    Raises
    ------

    ValueError
        If the backend choice is not 'numpy' or 'torch'.
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

    Parameters
    ----------

    volume : ArrayLike
        3D volume from which to generate the orthogonal planes.

    stride : tuple[int, int, int]
        Stride along the axes to generate the planes.

    Returns
    -------

    orthoplanes : tuple[ArrayLike]
        Tuple of 2D planes along the three canonical dimensions.
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

    Parameters
    ----------

    shape : Iterable[int]
        Iterable containing integer shape information to check for squareness.

    dims : tuple[int] | None
        Tuple of dimensions to check for squareness. Defaults to None,
        which means that all dimensions are checked.


    Returns
    -------

    is_square : bool
        Boolean indicating whether the shape is square along the indicated dimensions.
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
    ----------
    
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
    ----------
    
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
    """Compute the size of a slice along an axis."""
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

    Parameters
    ----------

    path : PathLike
        Path to the underlying raw data on the filesystem.

    internal_path : str
        Internal path to the data inside the storage container.

    phase : Literal['train', 'val']
        Phase of the dataset: training or validation.

    planestride : tuple[int, int, int]
        Stride along the three axes to generate the orthogonal planes.

    tileshape : TileShape | None, optional
        Desired tileshape. If `None`, the maximally available tile is selected.
        The actual orthoplanes are generated from within the tiles.

    reader_class : type[Reader] | None, optional
        Reader class to use for loading the raw data from disk.
        The reader class will be deduced from the file suffix if `None`.

    transformer : Callable | None, optional
        Transformer class/callable that is applied to the orthogonal
        plane elements before returning them.

    classlabel_mapping : dict[str, int] | None, optional
        Mapping of class names to integer labels.
        Required for training phase datasets.

        
    Attributes
    ----------

    path : PathLike
        Path to the underlying raw data on the filesystem.

    internal_path : str
        Internal path to the data inside the storage container.

    phase : Literal['train', 'val']
        Phase of the dataset: training or validation.

    planestride : tuple[int, int, int]
        Stride along the three axes to generate the orthogonal planes.

    transformer : Callable | None
        Transformer class/callable that is applied to the orthogonal planes
        before returning them.

    classlabel_mapping : dict[str, int] | None
        Mapping of class names to integer labels.

    volume : ndarray
        The entire volume data loaded from disk.

    fingerprint : dict
        Metadata fingerprint of the dataset.

    baseshape : tuple[int, int, int]
        Basal shape of the 3D volume data.

    tileshape : TileShape
        Shape of the 3D tiles. From these tiles the orthogonal planes are generated.

    tiles : list[TileSlice]
        List of 3-tuples of slice objects that select the tiles from the full volume.
    
    orthoplanes : list[ArrayLike]
        List of concatenated 2D orthogonal planes along the three
        canonical dimensions.
    
    label : int
        Integer label of the dataset instance.
    """
    def __init__(self,
                 path: PathLike,
                 internal_path: str,
                 phase: Literal['train', 'val'],
                 planestride: tuple[int, int, int],
                 tileshape: TileShape | None = None,
                 reader_class: type[Reader] | None = None,
                 transformer: Callable | None = None,
                 classlabel_mapping: dict[str, int] | None = None,
                 ) -> None:

        super().__init__()

        self.path = Path(path)
        self.phase = phase
        self.planestride = planestride
        self.transformer = transformer
        self.classlabel_mapping = classlabel_mapping
        self.internal_path = internal_path
        self.reader = self._init_reader(reader_class, path, internal_path)
        
        if self.phase in {'train', 'val'} and classlabel_mapping is None:
            raise RuntimeError(f'Phase \'{self.phase}\' dataset requires a '
                               f'classlabel mapping!')
        
        # load from underlying storage with metadata fingerprint
        self.volume = self.reader.load_data()
        self.fingerprint = self.reader.load_fingerprint()

        self.baseshape = get_spatial_shape(self.volume.shape)
        self.tileshape, self.tiles = self._generate_tiles(tileshape)
        self.orthoplanes = self._generate_orthoplanes()


    @staticmethod
    def _init_reader(reader_class: type[Reader] | None, path: PathLike, internal_path: str) -> Reader:
        if reader_class is None:
            reader_class = deduce_reader_class(path)
        return reader_class(path=path, internal_path=internal_path)
    
        
    def _generate_tiles(self,
                        tileshape: TileShape | None
                        ) -> tuple[TileShape, list[TileSlice]]:
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
            builder = VolumeTileBuilder(
                baseshape=self.baseshape, tileshape=tileshape,
                radius=radius
            )
            tiles = builder.tiles
        return (tileshape, tiles)
    
    
    def _generate_orthoplanes(self) -> list[ArrayLike]:
        """
        Generate a single large list of all orthogonal planes from all
        the tiles in the dataset.
        """
        orthoplanes = []
        for tile in self.tiles:
            subvolume = np.squeeze(self.volume[tile])
            orthoplanes.extend(
                generate_orthogonal_planes(volume=subvolume, stride=self.planestride)
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
        orthoplane = torch.tensor(orthoplane)

        if self.transformer:
            orthoplane = self.transformer(orthoplane)

        if self.phase == 'test':
            return orthoplane
        
        label = torch.tensor(self.label).unsqueeze_(-1)

        return (orthoplane, label)


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



class TriaxialDatasetBuilder:
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
              tileshape: TileShape,
              planestride: tuple[int, int, int],
              transform_configurations: Iterable[dict] | None = None,
              **kwargs
              ) -> list[TriaxialDataset]:
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
        try:
            fingerprint = cls.instance_mapping[ID]
        except KeyError:
            raise FileNotFoundError(f'could not retrieve dataset instance with ID "{ID}" - '
                                    f'check if ID is present in the data configuration!')

        return fingerprint.location