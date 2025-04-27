import logging

from collections.abc import Callable, Sequence
from typing import Literal
from pathlib import Path
from functools import cached_property

import attrs
import zarr
import numpy as np
import torch
import tqdm.auto as tqdm

from torch.utils.data import Dataset

from woodnet.datasets.fingerprints import Fingerprint, StatParams
from woodnet.transformations.transforms import Normalize
from woodnet.datasets.utils import generate_cylindrical_roi
from woodnet.datasets.summary.summary import scrape_directory
import woodnet.transformations.buildtools

from woodnet.datasets.planar.tilehelpers import compute_centroid_square, to_slices

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


class TileDataset(Dataset):

    def __init__(
        self,
        phase: Literal['train', 'val', 'test'],
        data: np.ndarray,
        fingerprint: Fingerprint,
        stats: StatParams,
        transformer: Callable[[torch.Tensor], torch.Tensor] = None,
        classlabel_mapping: dict[str, int] = None,
    ) -> None:
        """
        Initialize the PlaneDataset.

        Parameters
        ----------
        data : np.ndarray
            The data to be used in the dataset.
        """
        self.phase: Literal['train', 'val', 'test'] = phase
        self.shape = None
        self.channels: int = 0
        self.data: np.ndarray = self._initialize_data(data)
        self._log_initialization()
        self.transformer = transformer
        self.fingerprint: Fingerprint = fingerprint
        self.stats: StatParams = stats
        self.classlabel_mapping= classlabel_mapping or {}


    def _initialize_data(self, data: np.ndarray) -> np.ndarray:
        """Determine the data shape and add a fake channel dimension if needed."""
        if data.ndim == 3:
            self.shape = data.shape
            logger.debug(f'Adding fake channel dimension to data shape: {data.shape}')
            # add a fake channel dimension
            self.channels = 1
            return np.expand_dims(data, axis=0)
        elif data.ndim == 4:
            # data already has a channel dimension
            C, *shape = data.shape
            self.shape = shape
            self.channels = C
            return data

        raise ValueError(
            f'Invalid data shape: {data.shape}. Expected 3D or 4D array '
            f'with layout (C, D, H, W) or (D, H, W) respectively.'
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Get the plane item at the specified index.
        For training and validation, return the plane and label.
        For testing, return only the plane.
        """
        # select along vertical z axis - data layout (C x D x H x W)
        plane = torch.tensor(self.data[:, index, ...])
        # apply transformer if provided
        if self.transformer is not None:
            plane = self.transformer(plane)

        if self.phase == 'test':
            return plane
        label = torch.tensor(self.label).unsqueeze_(0)
        return (plane, label)


    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return self.shape[0]
    
    @property
    def planeshape(self) -> tuple[int, int]:
        return self.shape[1:]
    
    @property
    def volumeshape(self) -> tuple[int, int, int]:
        return self.shape

    @cached_property
    def class_(self) -> str:
        return self.fingerprint.class_

    @cached_property
    def label(self) -> int:
        return self.classlabel_mapping[self.fingerprint.class_]
    

    def _log_initialization(self) -> None:
        logger.info(
            f'{self.__class__.__name__} initialized with phase: {self.phase}, '
            f'data shape: {self.shape}, channels: {self.channels} and '
            f'total length: {len(self)}'
        )


class TileSelector:

    def __init__(
        self,
        baseshape: tuple[int, int, int],
        tileshape: tuple[int, int]
    ):
        self.baseshape = baseshape
        self.tileshape = tileshape


class BaseSubselector:
    def __call__(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement __call__ method.'
        )


class CentroidTileSubselector(BaseSubselector):
    """
    Subselect a square in-plane tile from the center of the data.
    Supports subselection along the z-axis.
    Input data is expected to be in the layout:
        ([...pre_dims...] x D x H x W)
    where D is the depth, H is the height and W is the width
    and an arbitrary number of pre-dimensions. 
    """
    log_action: bool = True
    def __init__(
        self,
        z_spacing: int | None = None,
    ) -> None:
        self.z_spacing = z_spacing

    def __call__(self, data: np.ndarray) -> np.ndarray:
        *pre, D, H, W = data.shape
        if self.z_spacing is not None:
            zslice = slice(0, D, self.z_spacing)
        else:
            zslice = slice(0, D)
        yx_slices = to_slices(*compute_centroid_square((H, W)))
        wildcards = tuple(slice(None) for _ in range(len(pre)))
        subselection = data[*wildcards, zslice, *yx_slices]
        if self.log_action:
            self._log_subselection(data, subselection)
        return subselection
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(z_spacing={self.z_spacing})'

    def __str__(self) -> str:
        return repr(self)

    def _log_subselection(self, input: np.ndarray, output: np.ndarray) -> None:
        logger.debug(
            f'{str(self)} subselection action: {input.shape} -> {output.shape}'
        )



def build_from_zarr(
    path: Path,
    internal_path: str,
    phase: Literal['train', 'val', 'test'],
    transformer: Callable[[torch.Tensor], torch.Tensor],
    subselector: BaseSubselector | Callable[[np.ndarray], np.ndarray] | None = None,
    scaling_policy: Literal['default', 'clipping'] | None = 'default',
    classlabel_mapping: dict[str, int] | None = None, 
) -> TileDataset:
    
    logger.debug(
        f'Starting build process for TileDataset from \'{path}{internal_path}\''
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

    if scaling_policy is not None:
        scaler_creator = get_scaler_function(scaling_policy)
        scaler = scaler_creator(path, internal_path)
        transformer.prepend(scaler)

    dataset = TileDataset(
        phase=phase,
        data=data,
        fingerprint=fingerprint,
        stats=stats,
        transformer=transformer,
        classlabel_mapping=classlabel_mapping,
    )
    return dataset


DEFAULT_CLASSLABEL_MAPPING: dict[str, int] = {
    'acer' : 0,
    'pinus' : 1
}


def bulk_build_from_zarr(
    base_directory: Path,
    IDs: Sequence[str],
    internal_path: str,
    phase: Literal['train', 'val', 'test'],
    scaling_policy: Literal['default', 'clipping'] = 'default',
    transform_configurations: Sequence[dict] | None = None,
    classlabel_mapping: dict[str, int] =  DEFAULT_CLASSLABEL_MAPPING,
) -> list[TileDataset]:
    transform_configurations = transform_configurations or []
    available_IDs = scrape_directory(base_directory)
    
    transformer = woodnet.transformations.buildtools.from_configurations(
        transform_configurations
    )
    

    datasets = []
    for ID in tqdm.tqdm(IDs):
        try:
            fpath = available_IDs[ID]
        except KeyError:
            logger.warning(
                f'Dataset with ID {ID} not found in base directory {base_directory}. '
                f'Skipping this ID for dataset build process.'
            )
            continue









def clip_and_recompute(data: np.ndarray, a_min: float, a_max: float) -> dict[str, float]:
    data = np.clip(data, a_min=a_min, a_max=a_max)
    mean = np.mean(data)
    stdev = np.std(data)
    return {'mean' : mean, 'stdev' : stdev}


@attrs.define
class ParameterChange:
    pre: float
    post: float
    change: float = attrs.field(init=False)
        
    def __attrs_post_init__(self) -> None:
        self.change = self.post / self.pre


def evaluate_parameter_change(
    zarray: zarr.Array
) -> dict[str, ParameterChange]:
    fingerprint = Fingerprint.from_zarr_array(zarray)
    statparams = StatParams.from_zarr_array(zarray)
    if fingerprint.roi == 'cylindrical-center':
        mask = generate_cylindrical_roi(zarray.shape)
    else:
        mask  = np.s_[...]
    data = zarray[...][mask]
    new_parameters = clip_and_recompute(data, a_min=statparams.q_05, a_max=statparams.q_95)
    mean_param = ParameterChange(pre=statparams.mean, post=new_parameters['mean'])
    stdev_param = ParameterChange(pre=statparams.stdev, post=new_parameters['stdev'])
    return {'mean' : mean_param, 'stdev' : stdev_param}


def create_clipping_scaler(
    path: Path,
    internal_path: str,      
) -> Normalize:
    """
    Create a scaler that clips the data to the precomputed 5th and 95th percentiles
    and uses the subsequent mean and standard deviation for normalization.
    """
    zarrobj = zarr.convenience.open(path, mode='r')
    stats = StatParams.from_zarr_array(zarrobj[internal_path])
    fingerprint = Fingerprint.from_zarr_array(zarrobj[internal_path])
    data = zarrobj[internal_path][...]
    if fingerprint.roi == 'cylindrical-center':
        mask = generate_cylindrical_roi(data.shape)
    else:
        mask = np.s_[...]
    data = data[mask]
    new_parameters = clip_and_recompute(data, a_min=stats.q_05, a_max=stats.q_95)
    new_mean = new_parameters['mean']
    new_stdev = new_parameters['stdev']
    logger.debug(
        f'Created clipping scaler with recomputed parameters: mean = {new_mean} and '
        f'{new_stdev} using ROI = \'{fingerprint.roi}\' from path \'{path}{internal_path}\''
    )
    return Normalize(mean=new_mean, std=new_stdev)


def create_default_scaler(
    path: Path,
    internal_path: str
) -> Normalize:
    """
    Create a default scaler for the dataset that normalizes to
    the global mean and standard deviation of the dataset.
    """
    zarrobj = zarr.convenience.open(path, mode='r')
    stats = StatParams.from_zarr_array(zarrobj[internal_path])
    logger.debug(
        f'Created default scaler with parameters: mean = {stats.mean} and '
        f'stdev = {stats.stdev} from path \'{path}{internal_path}\''
    )
    return Normalize(mean=stats.mean, std=stats.stdev)



def get_scaler_function(
    scaling_policy: Literal['default', 'clipping']
) -> Callable[[Path, str], Normalize]:
    """
    Get the scaler function based on the scaling policy.
    """
    if scaling_policy == 'default':
        return create_default_scaler
    elif scaling_policy == 'clipping':
        return create_clipping_scaler
    else:
        raise ValueError(
            f'Invalid scaling policy: {scaling_policy}. '
            f'Expected one of [\'default\', \'clipping\']'
        )