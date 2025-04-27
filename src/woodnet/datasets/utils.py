import numpy as np
from pathlib import Path

import skimage.morphology as morph

from woodnet.datasets.constants import (COMPLETE_DATASET_DIRECTORY,
                                        CLASS_ID_ORIENTATION_MAPPING)

def add_channel_dim(array: np.ndarray) -> np.ndarray:
    """Add fake channel dimension."""
    return array[np.newaxis, ...]


def get_spatial_shape(shape: tuple[int]) -> tuple[int]:
    """Get spatial shape for 4D inputs"""
    return shape[1:]


def retrieve_directory(ID: str) -> Path:
    for child in COMPLETE_DATASET_DIRECTORY.iterdir():
        if not child.is_dir():
            continue
        if child.match(f'{ID}_*'):
            return child
    raise FileNotFoundError(f'ID {ID} directory not found @ expected '
                            f'location: {COMPLETE_DATASET_DIRECTORY}')


def get_ID_by(class_: str, orientation: None | str = None) -> list[str]:
    IDs_mapping = CLASS_ID_ORIENTATION_MAPPING[class_]
    if orientation:
        IDs = [
            ID for ID, ostate in IDs_mapping.items() if ostate == orientation
        ]
    else:
        IDs = list(IDs_mapping.keys())
    return IDs


def compute_statistics(
    volume: np.ndarray,
    mask: np.ndarray | None = None
) -> dict:
    inshape = volume.shape
    voxelcount = volume.size
    if mask is not None:
        volume = volume[mask]
    statistics = {}
    # Cast to float to make the values JSON-serializable.
    # 32 bit numpy floats are not natively JSON-serializable
    statistics['minimum'] = float(np.min(volume))
    statistics['maximum'] = float(np.max(volume))
    statistics['mean'] = float(np.mean(volume))
    statistics['median'] = float(np.median(volume))
    statistics['stdev'] = float(np.std(volume))
    statistics['total_voxelcount'] = voxelcount
    statistics['roi_voxelcount'] = np.count_nonzero(mask) if mask is not None else voxelcount
    statistics['shape'] = inshape
    statistics['q_95']  = float(np.quantile(volume, q=0.95))
    statistics['q_05']  = float(np.quantile(volume, q=0.05))
    statistics['q_99']  = float(np.quantile(volume, q=0.99))
    statistics['q_01']  = float(np.quantile(volume, q=0.01))
    return statistics


def generate_cylindrical_roi(
    shape: tuple[int, ...],
    *,
    dtype: np.dtype = bool,
    ) -> np.ndarray:
    try:
        *pre, z_sz, y_sz, x_sz = shape
    except ValueError:
        pre = ()
        z_sz, y_sz, x_sz = shape
    if not y_sz == x_sz:
        raise ValueError(f'cannot create cylindrical ROI for non-square shape with {x_sz=} and {y_sz=}')
    
    disk = morph.disk(radius=x_sz//2)
    if disk.shape != (y_sz, x_sz):
        disk = disk[:-1, :-1]
    assert disk.shape == (y_sz, x_sz), 'wat'
    disk = np.broadcast_to(disk[np.newaxis, ...], shape=(z_sz, y_sz, x_sz))
    disk = np.expand_dims(disk, axis=tuple(i for i in range(len(pre)))).astype(dtype)
    return disk