import logging

from typing import Protocol, Any
from collections.abc import Sequence

import numpy as np

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger: logging.Logger = logging.getLogger(DEFAULT_LOGGER_NAME)


class ReinitializableDataset(Protocol):
    # dataset must have an at lease 3D data array
    data: np.ndarray
    def reinitialize(self, data: np.ndarray) -> 'ReinitializableDataset':
        """
        Reinitialize the dataset with the given raw data.
        This method should return a new instance of the dataset
        with the updated raw data.
        """
        pass


def heal_datasets_2D(
    datasets: Sequence[ReinitializableDataset],
    pad_mode: str = 'edge',
    tolerance: int = 5,
    constant_value: float = 0.0,
    **kwargs: Any,
) -> list[ReinitializableDataset]:
    """
    Heal the datasets to have the same in-plane shape by padding them to the maximum shape.
    Healing is only applied to the in-plane dimensions, i.e. ([...] x H x W).
    The tolerance is the maximum allowed difference in the in-plane dimensions,
    otherwse a `ValueError` is raised.

    Parameters
    ----------

    datasets : Sequence[ReinitializableDataset]
        The datasets to be healed.

    pad_mode : str
        The padding mode to be used. Default is 'reflect'.
        Possible values are:

        'constant' 
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.

    constant_value : float, optional
        The constant value to be used for padding if `pad_mode` is 'constant'.
        Default is 0.

    **kwargs : Any
        Additional keyword arguments for the padding function.
        See `np.pad` for more details.
    """
    shapes = np.array([dataset.shape for dataset in datasets])
    shapeset = {dataset.shape for dataset in datasets}
    if len(shapeset) == 1:
        logger.info(
            f'Healing not necessary - homogenous datasets shape: {shapeset.pop()}'
        )
        return datasets
    max_sizes = np.max(shapes, axis=0)
    min_sizes = np.min(shapes, axis=0)
    *pre_diff, z_diff, y_diff, x_diff = max_sizes - min_sizes
    logger.info(
        f'Got N={len(datasets)} datasets with shapeset {shapeset} and '
        f'max_x_shape_diff={x_diff} and max_y_shape_diff={y_diff}'
    )
    diffs  = np.array((y_diff, x_diff))
    if np.any(diffs > tolerance):
        raise ValueError(
            f'Dataset shapes differ by more than {tolerance} voxels: {diffs}'
        )

    *_, y_target_sz, x_target_sz = max_sizes

    healed_datasets: list[ReinitializableDataset] = []

    for dataset in datasets:
        rawdata = dataset.data
        assert isinstance(rawdata, np.ndarray), f'expected np.ndarray, but got {type(rawdata)}'
        *pre, z_sz, y_sz, x_sz = rawdata.shape
        if y_sz == y_target_sz and x_sz == x_target_sz:
            healed_datasets.append(dataset)
            continue

        # pad rawdata to target shape
        y_pre = (y_target_sz - y_sz) // 2
        y_post = (y_target_sz - y_sz) // 2 + (y_target_sz - y_sz) % 2
        x_pre = (x_target_sz - x_sz) // 2
        x_post = (x_target_sz - x_sz) // 2 + (x_target_sz - x_sz) % 2
        # no padding for pre-dimensions and for z-dimension
        padspec = tuple(
            [
            *((0, 0) for _ in range(len(pre))),
             (0, 0), (y_pre, y_post), (x_pre, x_post)
            ]
        )
        if pad_mode == 'constant':
            # np.pad allows `constant_values` only for constant mode
            kwargs['constant_values'] = constant_value

        healed_rawdata = np.pad(
            rawdata,
            pad_width=padspec,
            mode=pad_mode,
            **kwargs
        )
        logger.debug(
            f'healed dataset shape {rawdata.shape} -> {healed_rawdata.shape} '
            f'with padspec {padspec} and pad_mode={pad_mode}'
        )
        healed_dataset = dataset.reinitialize(healed_rawdata)
        healed_datasets.append(healed_dataset)

    return healed_datasets





def heal_datasets_3D(
    datasets: Sequence[ReinitializableDataset],
    pad_mode: str = 'edge',
    tolerance: int = 5,
    constant_value: float = 0.0,
    **kwargs: Any,
) -> list[ReinitializableDataset]:
    """
    Heal the datasets to have the same volume shape by padding them to the maximum shape.
    Healing is only applied to the spatial dimensions, i.e. ([...] x D x H x W).
    The tolerance is the maximum allowed difference in the in-plane dimensions,
    otherwse a `ValueError` is raised.

    Parameters
    ----------

    datasets : Sequence[ReinitializableDataset]
        The datasets to be healed.

    pad_mode : str
        The padding mode to be used. Default is 'reflect'.
        Possible values are:

        'constant' 
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the
            array edge value.
        'maximum'
            Pads with the maximum value of all or part of the
            vector along each axis.
        'mean'
            Pads with the mean value of all or part of the
            vector along each axis.
        'median'
            Pads with the median value of all or part of the
            vector along each axis.
        'minimum'
            Pads with the minimum value of all or part of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.

    constant_value : float, optional
        The constant value to be used for padding if `pad_mode` is 'constant'.
        Default is 0.

    **kwargs : Any
        Additional keyword arguments for the padding function.
        See `np.pad` for more details.
    """
    shapes = np.array([dataset.shape for dataset in datasets])
    shapeset = {dataset.shape for dataset in datasets}
    if len(shapeset) == 1:
        logger.info(
            f'Healing not necessary - homogenous datasets shape: {shapeset.pop()}'
        )
        return datasets
    max_sizes = np.max(shapes, axis=0)
    min_sizes = np.min(shapes, axis=0)
    *pre_diff, z_diff, y_diff, x_diff = max_sizes - min_sizes
    logger.info(
        f'Got N={len(datasets)} datasets with shapeset {shapeset} and '
        f'(max_z_shape_diff={z_diff}, max_y_shape_diff={y_diff}, max_x_shape_diff={x_diff})'
    )
    diffs  = np.array((z_diff, y_diff, x_diff))
    if np.any(diffs > tolerance):
        raise ValueError(
            f'Dataset shapes differ by more than {tolerance} voxels: {diffs}'
        )

    *_, z_target_sz, y_target_sz, x_target_sz = max_sizes

    healed_datasets: list[ReinitializableDataset] = []

    for dataset in datasets:
        rawdata = dataset.data
        assert isinstance(rawdata, np.ndarray), f'expected np.ndarray, but got {type(rawdata)}'
        *pre, z_sz, y_sz, x_sz = rawdata.shape
        if z_sz == z_target_sz and y_sz == y_target_sz and x_sz == x_target_sz:
            healed_datasets.append(dataset)
            continue

        # pad rawdata to target shape
        z_pre = (z_target_sz - z_sz) // 2
        z_post = (z_target_sz - z_sz) // 2 + (z_target_sz - z_sz) % 2
        y_pre = (y_target_sz - y_sz) // 2
        y_post = (y_target_sz - y_sz) // 2 + (y_target_sz - y_sz) % 2
        x_pre = (x_target_sz - x_sz) // 2
        x_post = (x_target_sz - x_sz) // 2 + (x_target_sz - x_sz) % 2
        # no padding for pre-dimensions and for z-dimension
        padspec = tuple(
            [
            *((0, 0) for _ in range(len(pre))),
             (z_pre, z_post), (y_pre, y_post), (x_pre, x_post)
            ]
        )
        if pad_mode == 'constant':
            # np.pad allows `constant_values` only for constant mode
            kwargs['constant_values'] = constant_value

        #TODO: Remove in prod: extensive logging due to hang in healing
        logger.debug(
            f'starting healing dataset {dataset.stats._ID} and '
            f'internal path {dataset.stats._internal_path} '
            f'with shape {rawdata.shape} '
            f'and padspec {padspec} and pad_mode={pad_mode} '
            f'and kwargs {kwargs}'            
        )

        healed_rawdata = np.pad(
            rawdata,
            pad_width=padspec,
            mode=pad_mode,
            **kwargs
        )
        logger.debug(
            f'healed dataset shape {rawdata.shape} -> {healed_rawdata.shape} '
            f'with padspec {padspec} and pad_mode={pad_mode}'
        )
        healed_dataset = dataset.reinitialize(healed_rawdata)
        healed_datasets.append(healed_dataset)

    return healed_datasets