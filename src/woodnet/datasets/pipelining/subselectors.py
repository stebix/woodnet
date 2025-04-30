import logging

from enum import Enum

import numpy as np

from woodnet.datasets.pipelining.base import BaseSubselector
from woodnet.datasets.planar.slicebased import compute_centroid_square, to_slices

DEFAULT_LOGGER_NAME: str = '.'.join('main', __name__)
logger: logging.Logger = logging.getLogger(DEFAULT_LOGGER_NAME)


class ZSpacingStrategy(Enum):
    TIGHT = 'tight'
    SPREAD = 'spread'


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
            self._emit_action_log(data, subselection)
        return subselection
    


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
            self._emit_action_log(data, subselection)
        return subselection
    


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
            self._emit_action_log(data, subselection)
        return subselection
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(z_spacing={self.z_spacing})'

    def __str__(self) -> str:
        return repr(self)