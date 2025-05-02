import logging

from enum import Enum

import numpy as np

from woodnet.datasets.pipelining.base import BaseSubselector
from woodnet.datasets.planar.slicebased import compute_centroid_square, to_slices
from woodnet.datasets.pipelining.arrayseqmap import map_func_to_arrays, ArraySequence

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger: logging.Logger = logging.getLogger(DEFAULT_LOGGER_NAME)


class ZSpacingStrategy(Enum):
    TIGHT = 'tight'
    SPREAD = 'spread'


class PhysicalCenterCubeSubselector(BaseSubselector):
    """
    This subselector selects a centroid cube from the input data.
    The cube edge lengths are fully determined based on the desired
    `target_in_plane_length` that specifies the *physical*
    size of the in-plane tile.
    In the top down view, the selection looks like this:
        *********
        *       *
        *  :::  *       Here * denotes the host volume boundaries
        *  : :  *       and : denotes the selected cube
        *  :::  *
        *       *
        *********
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
    
    def __call__(self, data: ArraySequence | np.ndarray) -> ArraySequence | np.ndarray:
        return map_func_to_arrays(data, self.apply_to)
    
    def apply_to(self, data: np.ndarray) -> np.ndarray:
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
            
    def __call__(self, data: ArraySequence | np.ndarray) -> ArraySequence | np.ndarray:
        return map_func_to_arrays(data, self.apply_to)
        
    def apply_to(self, data: np.ndarray) -> np.ndarray:
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
    E.g. from top down view, we select the square that is inscribed
    in the circle that is the salient foreground ROI.

         *******                        
      ***       ***                     
     *   .......   *                    
    *    :     :    *                   
    *    :     :    *    <- Inner cube is selected             
    *    :     :    *                   
     *   '''''''   *                    
      ***       ***                     
         *******       

    The subselector may yield multiple centroid cubes as a result
    when the in plane edge length `(s, s)` fits multiple times
    along the depth axis, i.e. `D // s > 1`.
    Input data is expected to be in the layout:
        ([...pre_dims...] x D x H x W)
    where D is the depth, H is the height and W is the width
    and an arbitrary number of pre-dimensions. 
    """
    def __init__(
        self,
        cube_limit: int | None = None,
    ) -> None:
        self.cube_limit = cube_limit

    def __call__(self, data: ArraySequence | np.ndarray) -> ArraySequence | np.ndarray:
        return map_func_to_arrays(data, self.apply_to)
        
    def apply_to(self, data: np.ndarray) -> np.ndarray | list[np.ndarray]:
        *pre, D, H, W = data.shape
        (sy, sx, s, s) = compute_centroid_square((H, W))
        yx_slices = to_slices(sy, sx, s ,s)
        zslices = self._compute_z_slices(
            D=D,
            size=s,
            max_cubes=self.cube_limit
        )
        wildcards = tuple(slice(None) for _ in range(len(pre)))
        subselection = []
        for zslice in zslices:
            subselection.append(data[*wildcards, zslice, *yx_slices])

        # squeeze the output if only one cube is selected
        if len(subselection) == 1:
            subselection = subselection[0]

        if self.log_action:
            self._emit_action_log(data, subselection)
            
        return subselection
    
    @staticmethod
    def _compute_z_slices(D: int, size: int, max_cubes: int | None) -> list[slice]:
        """
        Compute the z-slices for the centroid cubes of edge length `size`.
        The z slices are computed such that the cubes are distanced
        evenly along the z-axis.
        The number of cubes is limited to `max_cubes`.
        """
        count = min(D // size, max_cubes) if max_cubes is not None else D // size
        remainder = D - (count * size)
        if count == 0:
            raise ValueError(
                f'cannot fit cube of size {size} into data of depth {D}.'
            )
        zslices = []
        for i in range(count):
            start = i * size + (remainder // count) * i
            end = start + size
            zslices.append(slice(start, end))
        return zslices

    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(cube_limit={self.cube_limit})'

    def __str__(self) -> str:
        return repr(self)

    def _emit_action_log(
        self,
        input: np.ndarray,
        output: np.ndarray | list[np.ndarray]
    ) -> None:
        if isinstance(output, list):
            msg_part = f'N={len(output)} items with shape {output[0].shape}'
        else:
            msg_part = f' {output.shape}'
        logger.debug(
            f'{str(self)} subselection action: {input.shape} -> {msg_part}'
        )
