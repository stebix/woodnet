from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping

import attrs
import zarr

PathLike = str | Path


@attrs.define
class Fingerprint:
    """
    General dataset fingerprint information.
    """
    ID: str
    class_: str
    orientation_group: str
    processing_group: str
    # optional
    position_group: str | None = None
    current: float | None = None
    exposure: float | None = None
    voltage: float | None = None
    navg: int | None = None
    roi: str | None = None
    voxel_size: float | None = None
    scale_factor: float | None = None
    augmentation_specification: Mapping | None = None

    @classmethod
    def from_zarr(
        cls,
        file_or_path: zarr.Group | PathLike,
        internal_path: PathLike,
    ) -> 'Fingerprint':
        """
        Create a Fingerprint instance from a zarr file path or already group.

        Parameters
        ----------
        file_or_path : zarr.Group | PathLike
            The zarr file path or already opened zarr group.

        internal_path : PathLike
            The internal path to the zarr dataset.
            Note: This is relative to the initial point. If a group is passed,
            this should be the path to the array relative to the group.

        Returns
        -------
        Fingerprint
            An instance of Fingerprint with the fingerprint from the zarr group.
        """
        if isinstance(file_or_path, (str, Path)):
            zarrfile = zarr.convenience.open(file_or_path, mode='r')
        else:
            zarrfile = file_or_path
        info = deepcopy(dict(zarrfile[str(internal_path)].attrs))
        info.pop('statistics', None)
        return cls(**info)
    
    @classmethod
    def from_zarr_array(
        cls,
        array: zarr.Array,
    ) -> 'Fingerprint':
        """
        Create a Fingerprint instance directly from a zarr array.

        Parameters
        ----------
        array : zarr.Array
            The zarr array containing the fingerprint as metadata.

        Returns
        -------
        Fingerprint
            An instance of Fingerprint with the fingerprint from the zarr group.
        """
        info = deepcopy(dict(array.attrs))
        info.pop('statistics', None)
        return cls(**info)



@attrs.define
class StatParams:
    """
    Class to hold the statistics of a dataset.
    """
    mean: float
    stdev: float
    median: float
    minimum: float
    maximum: float
    q_01: float
    q_05: float
    q_95: float
    q_99: float
    shape: tuple[int, int, int]
    total_voxelcount: int | None = None
    roi_voxelcount: int | None = None
        
    _ID: str | None = attrs.field(alias='_ID', default=None, repr=False)
    _internal_path: str | None = attrs.field(alias='_internal_path', default=None, repr=False)

    @classmethod
    def from_zarr_array(
        cls,
        array: zarr.Array,
        statistics_key: str = 'statistics'
    ) -> 'StatParams':
        """
        Create a StatParams instance directly from a zarr array.

        Parameters
        ----------
        array : zarr.Array
            The zarr array containing the statistics as metadata.

        statistics_key : str, optional
            The key to access the statistics in the group. Default is 'statistics'.

        Returns
        -------
        StatParams
            An instance of StatParams with the statistics from the zarr group.
        """
        info = deepcopy(dict(array.attrs))
        ID = info['ID']
        kwargs = info[statistics_key]
        return cls(**kwargs, _ID=ID, _internal_path=array.path)


    @classmethod
    def from_zarr(
        cls,
        file_or_path: zarr.Group | PathLike,
        internal_path: PathLike,
        statistics_key: str = 'statistics'
    ) -> 'StatParams':
        """
        Create a StatParams instance from a zarr file path or already group.

        Parameters
        ----------
        file_or_path : zarr.Group | PathLike
            The zarr file path or already opened zarr group.

        internal_path : PathLike
            The internal path to the zarr dataset.
            Note: This is relative to the initial point. If a group is passed,
            this should be the path to the array relative to the group.

        Returns
        -------
        StatParams
            An instance of StatParams with the statistics from the zarr group.
        """
        if isinstance(file_or_path, (str, Path)):
            zarrfile = zarr.convenience.open(file_or_path, mode='r')
        else:
            zarrfile = file_or_path
        info = deepcopy(dict(zarrfile[str(internal_path)].attrs))
        ID = info['ID']
        kwargs = info[statistics_key]
        return cls(**kwargs, _ID=ID, _internal_path=internal_path)