"""
Implement interfaces for reading data in different formats from disk.

For now, we will focus on reading data from zarr and HDF5 files.
If we want to support other formats, we can add them here by implementing
the corresponding reader class.

@Jannik Stebani 2024
"""
import abc
from pathlib import Path
import logging

import numpy as np
import zarr
import h5py

DEFAULT_LOGGER_NAME = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

PathLike = Path | str


def read_fingerprint_from_zarr(path: PathLike) -> dict:
    """
    Read the fingerprint from a zarr file.

    Parameters:
    ----------

    path : Path to the zarr file.

    Returns
    -------

    dict : The fingerprint of the zarr file.
    """
    data = zarr.convenience.open(path, mode='r')
    fingerprint = {k : v for k, v in data.attrs.items()}
    return fingerprint


def read_fingerprint_from_hdf5(path: PathLike, internal_path: str) -> dict:
    """ Read the fingerprint from a HDF5 file.
    
    Parameters
    ----------
    path : Path to the HDF5 file.
    internal_path : Path to the internal HDF5 dataset.

    Returns
    -------

    dict : The fingerprint of the HDF5 file.
    """
    with h5py.File(path, 'r') as handle:
        fingerprint = {k : v for k, v in handle[internal_path].attrs.items()}
    return fingerprint



def read_data_from_zarr(path: PathLike, internal_path: str) -> np.ndarray:
    """
    Eagerly read full data array from a zarr file.

    Parameters
    ----------

    path : Path to the zarr file.

    internal_path : Path to the internal zarr dataset.

    Returns
    -------

    zarr.Array : The data from the zarr file.
    """
    data = zarr.convenience.open(path, mode='r')
    data = data[internal_path][...]
    return data


def read_data_from_hdf5(path: PathLike, internal_path: str) -> np.ndarray:
    """
    Eagerly read full data array from a HDF5 file.

    Parameters
    ----------

    path : Path to the HDF5 file.

    internal_path : Path to the internal HDF5 dataset.

    Returns
    -------

    np.ndarray : The data from the HDF5 file.
    """
    with h5py.File(path, mode='r') as handle:
        data = handle[internal_path][...]
    return data



class Reader(abc.ABC):
    """Abstract interface for reading data from different file formats."""
    @abc.abstractmethod
    def load_data(self):
        """Load data from the file."""
        pass

    @abc.abstractmethod
    def load_fingerprint(self):
        """Load the fingerprint from the file."""
        pass


class ZarrReader(Reader):
    """Reader for zarr files."""
    def __init__(self, path: PathLike, internal_path: str):
        self.path = path
        self.internal_path = internal_path

    def load_data(self) -> np.ndarray:
        logger.debug(f'loading data from zarr file: {self.path}')
        return read_data_from_zarr(self.path, self.internal_path)

    def load_fingerprint(self) -> dict:
        logger.debug(f'loading fingerprint from zarr file: {self.path}')
        return read_fingerprint_from_zarr(self.path)



class HDF5Reader(Reader):
    """Reader for HDF5 files."""
    def __init__(self, path: PathLike, internal_path: str):
        self.path = path
        self.internal_path = internal_path

    def load_data(self) -> np.ndarray:
        logger.debug(f'loading data from HDF5 file: {self.path}')
        return read_data_from_hdf5(self.path, self.internal_path)

    def load_fingerprint(self) -> dict:
        logger.debug(f'loading fingerprint from HDF5 file: {self.path}')
        return read_fingerprint_from_hdf5(self.path, self.internal_path)




def extract_suffix(path: PathLike) -> str:
    """
    Extract the suffix from a path.

    Parameters
    ----------

    path : Path to the file.

    Returns
    -------

    str : The suffix of the file.
    """
    if isinstance(path, Path):
        return path.suffix
    elif isinstance(path, str):
        if '/' in path:
            return path.split('/')[-1].split('.')[-1]
        else:
            return path.split('.')[-1]
    
    raise TypeError(f'Invalid type: must be Path or str but got {type(path)}') 



def deduce_reader_class(path: PathLike) -> Reader:
    """
    Deduce the reader class (corresponding to the file format) from the path.
    """
    suffix = extract_suffix(path)
    if suffix.endswith('zarr'):
        return ZarrReader
    elif suffix.endswith('h5') or suffix.endswith('hdf5'):
        return HDF5Reader
    else:
        raise ValueError(f'Unsupported file format: {suffix}')


if __name__ == '__main__':
    path = Path('/home/jannik/storage/wood/custom/CT10.zarr')
    fingerprint = read_fingerprint_from_zarr(path)
    data = read_data_from_zarr(path, 'downsampled/half')
    print(data.shape)