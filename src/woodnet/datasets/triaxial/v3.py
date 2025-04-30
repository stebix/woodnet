import logging

from collections.abc import Callable, Sequence
from typing import TypeAlias, Literal
from itertools import product
from functools import cached_property
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import zarr
import tqdm.auto as tqdm

from woodnet.datasets.fingerprints import Fingerprint, StatParams
from woodnet.datasets.triaxial.typespecs import Planespec3D
import woodnet.datasets.planar.slicebased as slb


Dataset: TypeAlias = torch.utils.data.Dataset

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

def generate_plane_indices(
    shape: Sequence[int],
    axis: int,
    stride: int
) -> np.ndarray:
    """
    Generate a list of plane indices for a given axis and stride.
    Planes are given as start-stop parameters for slice objects.
    Layout is:
       [axis_0_start, axis_0_stop, axis_1_start, axis_1_stop, ..., axis_N_start, axis_N_stop]
    
    """
    dtype = np.uint16
    plane_indices = [i for i in range(0, shape[axis], stride)]
    pre_axis_parameters = np.array([(0, s) for s in shape[:axis]], dtype=dtype)
    post_axis_parameters = np.array([(0, s) for s in shape[axis+1:]], dtype=dtype)
    cache = []
    for index in plane_indices:
        indices = np.array((index, index), dtype=dtype)
        cache.append(
            np.concatenate((pre_axis_parameters.reshape(-1), indices, post_axis_parameters.reshape(-1)))
        )
    return np.array(cache)


def generate_triaxial_planes_parameter(
    shape: tuple[int, int, int],
    planestride: tuple[int, int, int],
) -> np.ndarray:
    """
    Generate a slice parameter for the given shape and planestride.
    """
    parameters = []
    for axis, stride in enumerate(planestride):
        parameters.append(
            generate_plane_indices(shape, axis, stride)
        )
    parameters = np.concatenate(parameters, axis=0)
    return parameters


def row_outer_product(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """
    Creates all combinations of rows from multiple 2D arrays.
    
    Parameters:
    -----------
    arrays : list of numpy.ndarray
        List of 2D arrays whose rows will be combined
    
    Returns:
    --------
    numpy.ndarray
        3D array where:
        - First dimension represents each combination
        - Second dimension represents each input array
        - Third dimension contains the actual row data
        
    Example:
    --------
    For arrays A(n1, d1), B(n2, d2), C(n3, d3):
    Result shape will be (n1*n2*n3, 3, max(d1,d2,d3))
    Where result[i, 0, :] gives the row from A for combination i
          result[i, 1, :] gives the row from B for combination i
          result[i, 2, :] gives the row from C for combination i
    """
    shapes = [array.shape for array in arrays]
    if any(len(shape) > 2 for shape in shapes):
        raise ValueError(
            f'all arrays must be 2D, but got shapes: {set(s for s in shapes)}'
        )
    trailing_axis_sizes = {shape[-1] for shape in shapes}
    if len(trailing_axis_sizes) > 1:   
        raise ValueError(
            f'rows of input arrays must have same size, but mutliple trailing'
            f'axis sizes: {trailing_axis_sizes}'
        )
    
    dtype = np.uint16
    num_arrays = len(arrays)
    # Calculate the number of combinations
    num_combinations = np.prod([shape[0] for shape in shapes])
    row_size = trailing_axis_sizes.pop()
    # Create output array with appropriate shape
    result = np.full(
        shape=(num_combinations, num_arrays, row_size), fill_value=-1, dtype=dtype
    )
    # Get all indices combinations
    indices = list(product(*[range(shape[0]) for shape in shapes]))
    
    for i, idx_tuple in enumerate(indices):
        for j, (array_idx, array) in enumerate(zip(idx_tuple, arrays)):
            # Copy the actual data - missing data will be -1
            row_length = arrays[j].shape[1]
            result[i, j, :row_length] = arrays[j][array_idx]
    
    return result


def convert_to_slices_from(parameters: Sequence[int]) -> tuple[Planespec3D, ...]:
    """
    Convert a list of start-stop parameters to a tuple of slice objects.
    If `start == stop`, we interpret it as an axis wildcard slice `slice(None)`.
    """
    slices = []
    for i in range(0, len(parameters), 2):
        start = parameters[i]
        stop = parameters[i + 1]
        slc = slice(start, stop) if start != stop else start
        slices.append(slc)
    return tuple(slices)


def compute_array_memory_size(
    array: np.ndarray,
    unit: Literal['KB', 'MB', 'GB'] = 'MB'
) -> float:
    """
    Compute the memory size of a numpy array in bytes.
    """
    unit_factor: dict[str, float] = {
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3
    }
    try:
        factor = unit_factor[unit]
    except KeyError:
        raise ValueError(f'Invalid unit: {unit}. Use one of {list(unit_factor.keys())}.')
    return array.nbytes / factor


class V3TriaxialDataset(Dataset):
    """
    Latest addition to the triaxial dataset family.
    """
    def __init__(
        self,
        phase: Literal['train', 'val', 'test'],
        data: np.ndarray,
        planestride: tuple[int, int, int],
        fingerprint: Fingerprint,
        stats: StatParams,
        transformer: Callable[[torch.Tensor], torch.Tensor] = None,
        classlabel_mapping: dict[str, int] = None,
    ) -> None:
        """
        Initialize the dataset with the given parameters.
        """
        self.phase: Literal['train', 'val', 'test'] = phase
        self.shape = None
        self.channels: int = 0
        self.data: np.ndarray = self._initialize_data(data)
        self.planestride = planestride
        self.transformer = transformer
        self.fingerprint: Fingerprint = fingerprint
        self.stats: StatParams = stats
        self.classlabel_mapping = classlabel_mapping or {}
        self.triaxial_plane_parameters = self._build_triax_planes_specs()


    def _initialize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Determine the data shape and add a fake channel dimension if needed.
        In this dataset we require 3D data - the channels are the concatenated
        triaxial planes - thus `C` is always 3.
        """
        if len(data.shape) != 3:
            raise ValueError(
                f'{self.__class__.__name__} input data must be 3D, but got {len(data.shape)}D data'
            )
        self.channels = 3
        self.shape = data.shape
        return data


    def _build_triax_planes_specs(self) -> np.ndarray:
        """
        Build the numpy index array spcifying the triaxial planes.
        """
        print(self.shape)
        print(self.planestride)
        # this gives us the planes separate for every axis
        axiswise_parameters = [
            generate_plane_indices(self.shape, axis, stride)
            for axis, stride in enumerate(self.planestride)
        ]
        # we still need to combine - this is done via the row-wise outer product
        triax_planes_parameters = row_outer_product(axiswise_parameters)
        assert triax_planes_parameters.shape[1] == 3, 'expecting 3 planes'
        assert triax_planes_parameters.shape[2] == 6, 'expecting 2*3=6 parameters per plane'
        logger.debug(
            f'{self.__class__.__name__} generated triaxial planes parameters with shape: '
            f'{triax_planes_parameters.shape} for a base volume shape of {self.shape} '
            f'and planestride of {self.planestride}. Triaxial planes parameters array size: '
            f'{compute_array_memory_size(triax_planes_parameters, unit="MB"):.2f} MB'
        )
        return triax_planes_parameters


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset.
        """
        plane_parameters = self.triaxial_plane_parameters[index]
        planespec = np.apply_along_axis(
            convert_to_slices_from,
            axis=1,
            arr=plane_parameters
        )
        zplane, yplane, xplane = tuple(tuple(p) for p in planespec)
        planes = np.stack(
            (self.data[zplane], self.data[yplane], self.data[xplane]),
            axis=0
        )
        planes = torch.tensor(planes)
        if self.transformer:
            planes = self.transformer(planes)

        if self.phase == 'test':
            return planes
        
        label = torch.tensor(self.label).unsqueeze(0)
        return (planes, label)

    def __len__(self) -> int:
        # expected layout is (i_combination, i_plane, i_parameter)
        return self.triaxial_plane_parameters.shape[0]


    @property
    def volumeshape(self) -> tuple[int, int, int]:
        return self.shape

    @cached_property
    def class_(self) -> str:
        return self.fingerprint.class_

    @cached_property
    def label(self) -> int:
        return self.classlabel_mapping[self.fingerprint.class_]


    @classmethod
    def build_from_zarr(
        cls,
        path: str | Path,
        internal_path: str,
        phase: Literal['train', 'val', 'test'],
        planestride: tuple[int, int, int],
        transformer: Callable[[torch.Tensor], torch.Tensor],
        subselector: slb.BaseSubselector | Callable[[np.ndarray], np.ndarray] | None = None,
        scaling_policy: Literal['default', 'clipping'] | None = 'default',
        classlabel_mapping: dict[str, int] | None = None, 
    ) -> 'V3TriaxialDataset':
        """
        Build a dataset from a zarr file.
        """
        logger.debug(
            f'Starting build process for {cls.__name__} from \'{path}{internal_path}\''
        )
        if classlabel_mapping is None and phase != 'test':
            raise ValueError(
                'classlabel_mapping must be provided for training and validation phases'
            )
        
        zarrobj = zarr.convenience.open(path, mode='r')
        stats = StatParams.from_zarr_array(zarrobj[internal_path])
        fingerprint = Fingerprint.from_zarr_array(zarrobj[internal_path])
        data = zarrobj[internal_path][...]

        if subselector is not None:
            logger.debug(
                f'Applying subselector {str(subselector)} to data with shape {data.shape} '
                f'from source \'{path}{internal_path}\''
            )
            data = subselector(data)

        # avoid global state via list
        transformer = deepcopy(transformer)

        if scaling_policy is not None:
            scaler_creator = slb.get_scaler_function(scaling_policy)
            scaler = scaler_creator(path, internal_path)
            transformer.prepend(scaler)

        dataset = cls(
            phase=phase,
            data=data,
            planestride=planestride,
            fingerprint=fingerprint,
            stats=stats,
            transformer=transformer,
            classlabel_mapping=classlabel_mapping,
        )
        return dataset

    @classmethod
    def bulk_build_from_zarr(
        cls,
        paths: Sequence[str | Path],
        internal_path: str,
        phase: Literal['train', 'val', 'test'],
        planestride: tuple[int, int, int],
        transformer: Callable[[torch.Tensor], torch.Tensor],
        subselector: slb.BaseSubselector | Callable[[np.ndarray], np.ndarray] | None = None,
        scaling_policy: Literal['default', 'clipping'] | None = 'default',
        classlabel_mapping: dict[str, int] | None = None, 
        leave_pbar: bool = False,
        pbar_desc: str | None = None
    ) -> list['V3TriaxialDataset']:
        """
        Bulk construct multiple datasets from paths. 
        """
        paths = [Path(p) if not isinstance(p, Path) else p for p in paths]
        datasets = []
        pbar_desc = pbar_desc or 'Dataset build progress'
        wrapped_paths = tqdm.tqdm(
            paths, unit='dset', leave=leave_pbar, desc=pbar_desc
        )
        for path in wrapped_paths:
            wrapped_paths.set_postfix_str(f'Loading {path.stem}')
            dataset = cls.build_from_zarr(
                path=path,
                internal_path=internal_path,
                phase=phase,
                planestride=planestride,
                transformer=transformer,
                subselector=subselector,
                scaling_policy=scaling_policy,
                classlabel_mapping=classlabel_mapping
            )
            datasets.append(dataset)
        return datasets