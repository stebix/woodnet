import logging

from collections.abc import Callable, Sequence, Mapping
from enum import Enum
from typing import Literal, Any
from pathlib import Path
from functools import cached_property
from copy import deepcopy

import attrs
import zarr
import numpy as np
import torch
import tqdm.auto as tqdm

from torch.utils.data import Dataset, DataLoader, ConcatDataset

from woodnet.datasets.fingerprints import Fingerprint, StatParams
from woodnet.transformations.transforms import Normalize
from woodnet.datasets.utils import generate_cylindrical_roi
from woodnet.datasets.summary.summary import scrape_directory
import woodnet.transformations.buildtools
from woodnet.transformations.transformer import Transformer
from woodnet.datasets.healing import heal_datasets_2D

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
        self.transformer = transformer
        self.fingerprint: Fingerprint = fingerprint
        self.stats: StatParams = stats
        self.classlabel_mapping= classlabel_mapping or {}
        self._log_initialization()


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
            f'Initialization success for {self.__class__.__name__} with phase=\'{self.phase}\' '
            f'shape={self.shape} channels={self.channels} and '
            f'length={len(self)} and class_=\'{self.class_}\' and label={self.label} '
        )

    def reinitialize(self, data: np.ndarray) -> 'TileDataset':
        """
        Reinitialize the dataset with new data.
        Intended usage:
        Healing dataset shape to enable batch collation after the multiple
        datasets have been created progrmmatically.
        We provide a separate method to avoid 'dirty' mutation of the
        data attribute.
        """
        return TileDataset(
            phase=self.phase,
            data=data,
            fingerprint=self.fingerprint,
            stats=self.stats,
            transformer=self.transformer,
            classlabel_mapping=self.classlabel_mapping,
        )
        


class BaseSubselector:
    log_action: bool = True

    def __call__(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement __call__ method.'
        )

    def _log_subselection(self, input: np.ndarray, output: np.ndarray) -> None:
        logger.debug(
            f'{str(self)} subselection action: {input.shape} -> {output.shape}'
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


class CentroidCubeSubselector(BaseSubselector):
    """
    Subselect a square in-plane tile from the center of the data
    that is assumed to be inside a enclosing circle.

    Here the 

    Input data is expected to be in the layout:
        ([...pre_dims...] x D x H x W)
    where D is the depth, H is the height and W is the width
    and an arbitrary number of pre-dimensions. 
    """
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





class ZSpacingStrategy(Enum):
    TIGHT = 'tight'
    SPREAD = 'spread'


class PhysicalCenterTileSubselector(BaseSubselector):
    """
    This subselector selects a centroid cuboid from the input data.

    The in-plane shape is determined based on the desired `target_in_plane_length`
    that specifies the *physical* size of the in-plane tile.
    The in-plane shape is computed based on the input voxel size.
    Along the z-axis, we can select the desired number of slices.
    The selection along the z-axis has two strategies:
    - TIGHT: densely select the z-slices around the center of the data
    - SPREAD: select the z-slices evenly distributed across full z-axis
    """
    def __init__(
        self,
        target_in_plane_length: float,
        input_voxel_size: float,
        target_slice_count: int,
        z_spacing_strategy: str | ZSpacingStrategy = ZSpacingStrategy.TIGHT,
    ) -> None:
        self.target_in_plane_length = target_in_plane_length
        self.input_voxel_size = input_voxel_size
        self.target_slice_count = target_slice_count
        self._in_plane_shape = self._compute_in_plane_shape(
            target_length=target_in_plane_length,
            voxel_size=input_voxel_size
        )
        if not isinstance(z_spacing_strategy, ZSpacingStrategy):
            z_spacing_strategy = ZSpacingStrategy(z_spacing_strategy)    
        self.z_spacing_strategy = z_spacing_strategy

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}(target_in_plane_length={self.target_in_plane_length}, '
                f'input_voxel_size={self.input_voxel_size}, '
                f'target_slice_count={self.target_slice_count}, '
                f'z_spacing_strategy={self.z_spacing_strategy})')

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def _compute_in_plane_shape(target_length: float, voxel_size: float) -> tuple[int, int]:
        """
        Compute the in-plane shape based on the desired target length
        and the voxel size of the data.
        """
        s = int(np.floor(target_length / voxel_size))
        return (s, s)
    
    @staticmethod
    def _compute_center_slices(H: int, W: int, tileshape: tuple[int, int]) -> tuple[slice, slice]:
        cy = H // 2
        cx = W // 2
        dy, dx = tileshape
        yslice = slice(cy - dy // 2, cy + dy // 2 + dy % 2)
        xslice = slice(cx - dx // 2, cx + dx // 2 + dx % 2)
        return (yslice, xslice)

    @staticmethod
    def _compute_z_indices(
        D: int,
        target_slice_count: int,
        z_spacing_strategy: ZSpacingStrategy
    ) -> np.ndarray:
        cz = D // 2
        if z_spacing_strategy is ZSpacingStrategy.TIGHT:
            z_indices = np.arange(
                cz - target_slice_count // 2,
                cz + target_slice_count // 2 + target_slice_count % 2
            )
        elif z_spacing_strategy is ZSpacingStrategy.SPREAD:
            z_indices = np.linspace(0, D, num=target_slice_count).astype(int)
            # protect against out of bounds and multi-selection of a slice
            z_indices = np.clip(z_indices, 0, D - 1)
            z_indices = np.unique(z_indices)
        else:
            raise ValueError(
                f'Invalid z_spacing_strategy: {z_spacing_strategy}. '
                f'Expected one of {ZSpacingStrategy.__members__}'
            )
        return z_indices
            

    def __call__(self, data: np.ndarray) -> np.ndarray:
        *pre, D, H, W = data.shape
        if self.target_slice_count > D:
            raise ValueError(
                f'requested target slice count {self.target_slice_count} is greater than data z size {D}.'
            )
        (yslice, xslice) = self._compute_center_slices(H, W, self._in_plane_shape)
        z_indices = self._compute_z_indices(
            D=D, target_slice_count=self.target_slice_count, z_spacing_strategy=self.z_spacing_strategy
        )
        wildcards = tuple(slice(None) for _ in range(len(pre)))
        subselection = data[*wildcards, z_indices, yslice, xslice]
        if self.log_action:
            self._log_subselection(data, subselection)
        return subselection



class PhysicalCenterCubeSubselector(BaseSubselector):
    """
    This subselector selects a centroid cube from the input data.
    The cube edge lengths are fully determined based on the desired
    `target_in_plane_length` that specifies the *physical* size of the in-plane tile.
    """
    def __init__(
        self,
        target_in_plane_length: float,
        input_voxel_size: float,
    ) -> None:
        self.target_in_plane_length = target_in_plane_length
        self.input_voxel_size = input_voxel_size
        self._cube_shape = self._compute_cube_shape(
            target_length=target_in_plane_length,
            voxel_size=input_voxel_size
        )

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}(target_in_plane_length={self.target_in_plane_length}, '
                f'input_voxel_size={self.input_voxel_size})')

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def _compute_cube_shape(target_length: float, voxel_size: float) -> tuple[int, int, int]:
        """
        Compute the cube shape based on the desired target length
        and the voxel size of the data.
        """
        s = int(np.floor(target_length / voxel_size))
        return (s, s, s)
    
    @staticmethod
    def _compute_center_slices(
        D: int, H: int, W: int,
        cubeshape: tuple[int, int, int]
    ) -> tuple[slice, slice, slice]:
        cz = D // 2
        cy = H // 2
        cx = W // 2
        dz, dy, dx = cubeshape
        zslice = slice(cz - dz // 2, cz + dz // 2 + dz % 2)
        yslice = slice(cy - dy // 2, cy + dy // 2 + dy % 2)
        xslice = slice(cx - dx // 2, cx + dx // 2 + dx % 2)
        return (zslice, yslice, xslice)
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        *pre, D, H, W = data.shape
        (zslice, yslice, xslice) = self._compute_center_slices(D, H, W, self._cube_shape)
        wildcards = tuple(slice(None) for _ in range(len(pre)))
        subselection = data[*wildcards, zslice, yslice, xslice]
        if self.log_action:
            self._log_subselection(data, subselection)
        return subselection


class ChannelSqueezingProcessor(BaseSubselector):
    """
    Subselector that squeezes the channel dimension of the data.
    """
    def __self__(
        self,
        multichannel_strategy: Literal['raise', 'warn', 'select'] = 'raise',
        channel_selection: int | None = None, 
    ) -> None:
        self.multichannel_strategy = multichannel_strategy
        self.channel_selection = channel_selection

    def __call__(self, data: np.ndarray) -> np.ndarray:
        *pre, C, D, H, W = data.shape
        pre = tuple(slice(None) for _ in range(len(pre)))
        if C == 1:
            processed = data[*pre, 0, :, :, :]
            if self.log_action:
                self._log_processing(data, processed, cidx=0)
            return processed
        if C > 1:
            if self.multichannel_strategy == 'raise':
                raise ValueError(
                    f'Input data has {C} channels, but only one channel is expected. '
                    f'Use multichannel_strategy=\'select\' to select a channel.'
                )
            elif self.multichannel_strategy == 'warn':
                logger.warning(
                    f'Input data has {C} channels, but only one channel is expected. '
                    f'Using channel index 0'
                )
                c_idx = 0
            elif self.multichannel_strategy == 'select':
                if self.channel_selection is None:
                    raise ValueError(
                        'channel_selection must be provided when multichannel_strategy=\'select\''
                    )
                c_idx = self.channel_selection
            
            processed = data[*pre, c_idx, :, :, :]
            if self.log_action:
                self._log_processing(data, processed, cidx=c_idx)
            return processed


    def _log_processing(self, input: np.ndarray, output: np.ndarray, cidx: int) -> None:
        logger.debug(
            f'{str(self)} channel processing action: {input.shape} -> {output.shape} '
            f'with channel index {cidx}'
        )


class Pipeline:

    def __init__(
        self,
        *steps: BaseSubselector,
    ) -> None:
        self.steps = steps

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the pipeline to the data.
        """
        for step in self.steps:
            data = step(data)
        return data



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

    # avoid global state via list
    transformer = deepcopy(transformer)

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
    paths: Sequence[Path],
    internal_path: str,
    phase: Literal['train', 'val', 'test'],
    transformer: Transformer | Callable[[torch.Tensor], torch.Tensor],
    subselector: BaseSubselector | Callable[[np.ndarray], np.ndarray] | None = None,
    scaling_policy: Literal['default', 'clipping'] | None = 'default',
    classlabel_mapping: dict[str, int] =  DEFAULT_CLASSLABEL_MAPPING,
    leave_pbar: bool = True,
) -> list[TileDataset]:
    
    datasets = []
    wrapped_paths = tqdm.tqdm(paths, desc='Building datasets', leave=leave_pbar)
    for path in wrapped_paths:
        wrapped_paths.set_postfix_str(f'Current: \'{path.stem}\'')
        dataset = build_from_zarr(
            path=path,
            internal_path=internal_path,
            phase=phase,
            subselector=subselector,
            transformer=transformer,
            scaling_policy=scaling_policy,
            classlabel_mapping=classlabel_mapping
        )
        datasets.append(dataset)

    return datasets



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


def bulk_group_build_from_zarr(
    fpaths: Sequence[Path],
    internal_paths: Sequence[str],
    phase: Literal['train', 'val', 'test'],
    transformer: Transformer | Callable[[torch.Tensor], torch.Tensor],
    subselector: BaseSubselector | Callable[[np.ndarray], np.ndarray] | None = None,
    scaling_policy: Literal['default', 'clipping'] | None = 'default',
    classlabel_mapping: dict[str, int] =  DEFAULT_CLASSLABEL_MAPPING,
    heal_in_group: bool = True,
    heal_kwargs: dict[str, Any] | None = None,
    leave_pbar: bool = True,
) -> dict[str, dict[str, TileDataset]]:
    """
    Helper function to create multiple groups datasets from 
    multiple internal paths at the Zarr stores. 
    """
    multisets: dict[str, dict[str, TileDataset]] = {}
    wrapped_internal_paths = tqdm.tqdm(
        internal_paths,
        desc='Group progress',
        leave=leave_pbar,
        unit='group'
    )
    for internal_path in wrapped_internal_paths:
        wrapped_internal_paths.set_postfix_str(f'Current: \'{internal_path}\'')
        datasets = bulk_build_from_zarr(
            paths=fpaths,
            internal_path=internal_path,
            phase=phase,
            transformer=transformer,
            subselector=subselector,
            scaling_policy=scaling_policy,
            classlabel_mapping=classlabel_mapping
        )
        if heal_in_group:
            heal_kwargs = heal_kwargs or {}
            datasets = heal_datasets_2D(
                datasets=datasets,
                **heal_kwargs
            )
        datasets = {
            dataset.fingerprint.ID : dataset
            for dataset in datasets
        }
        multisets[internal_path] = datasets
    return multisets



def multisets_to_loaders(
    multisets: dict[str, dict[str, TileDataset]],
    batch_size: int | Mapping[str, int],
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
    **kwargs: Any,
) -> dict[str, DataLoader]:
    """
    Create DataLoaders for each group of datasets in the multisets.
    Gist:
    - a group is a dictionary of datasets with the dataset ID as key
        i.e. a value of the top level `multisets` dictionary
    - a group will be merged into a single ConcatDataset
    - a DataLoader will be created from each ConcatDataset
    - the DataLoader will be returned in a dictionary with the group name as key
        i.e. the key of the top level `multisets` dictionary
    - the batch size can be a single integer or a dictionary with
      the group name as key to set the batch size group specific
        
    Parameters
    ----------
    multisets : dict[str, dict[str, TileDataset]]
        The multisets/groups of datasets to create DataLoaders for.
        Every group is a dictionary of datasets with the dataset ID as key.
        A group willt be merged into a single ConcatDataset.
    """
    loaders = {}
    if isinstance(batch_size, int):
        batch_size = {k : batch_size for k in multisets.keys()}
    batch_size = batch_size if isinstance(batch_size, Mapping) else {k: batch_size for k in multisets.keys()}
    for group, datasets in multisets.items():
        dataset = ConcatDataset(datasets=datasets.values())
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size[group],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs
        )
        loaders[group] = loader
    return loaders