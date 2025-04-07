"""
Tiling utilities for extracting subvolumes from cuboidal volumes.

In contrast to the cylindrical tiling, the cuboidal tiling can in-principle
use the entire volume for tiling.

Concept: Standoffs
------------------
Standoffs are used to create a safety margin around the volume
which is to be tiled. This is useful for example when the volume has artefacts
or other unwanted structures at the edges. The standoff can be specified
as fractions of the axis size (relative) or as absolute values (absolute).
The standoff is specified in the order (z, y, x).
If the value inside a standoff specification tuple is None, the corresponding
axis will not have a standoff.

@Author: Jannik Stebani 2025
"""
from collections.abc import Sequence
from itertools import product
from typing import NamedTuple, TypeAlias

import numpy as np


class Standoff(NamedTuple):
    z: float | int | None
    y: float | int | None
    x: float | int | None

Tilespec3D: TypeAlias = tuple[slice, slice, slice]
TilespecND: TypeAlias = tuple[slice, ...]
Tileshape3D: TypeAlias = tuple[int, int, int]

RelStandoff: TypeAlias = tuple[float | None, float | None, float | None]
AbsStandoff: TypeAlias = tuple[int | None, int | None, int | None]


def _compute_tiles_cuboidal(
    volshape: tuple[int, int, int],
    tileshape: tuple[int, int, int]
) -> tuple[Tilespec3D, ...]:
    """
    Compute the tilespecs for tiles of the given shape filling the
    cuboidal volume with the given `volshape`.

    Parameters
    ----------
    volshape : tuple[int, int, int]
        Shape of the volume to be tiled.
        The order is (z, y, x).
    
    tileshape : tuple[int, int, int]
        Shape of the tile to be used.
        The order is (z, y, x).

    Returns
    -------
    tuple[Tilespec3D, ...]
        Tilespecs for the tiles filling the volume.

    Raises
    ------
    ValueError
        If the tile shape cannot fit into the volume shape.
    """
    axis_wise_slices = []
    
    for i, (volax_sz, tilax_sz) in enumerate(zip(volshape, tileshape, strict=True)):
        # available repeats of tile along the axis
        n_reps = volax_sz // tilax_sz
        # available remainder space that can be distributed between the tiles
        remainder_total = volax_sz % tilax_sz
        if n_reps == 0:
            msg = (f'cannot fit tile with axis size {tilax_sz} into volume '
                   f'with axis size {volax_sz} - offending axis: {i}')
            raise ValueError(msg)
        elif n_reps == 1:
            remainder_increment = 0
        else:
            remainder_increment = int(remainder_total / (n_reps - 1))
        
        axis_slices = []
        start = 0
        for _ in range(n_reps): 
            stop = start + tilax_sz
            axis_slices.append(slice(start, stop))
            start = stop + remainder_increment
        
        axis_wise_slices.append((i, tuple(axis_slices)))
    
    # reformat to 3-tuples of slice objects the define a single subchunk
    (zi, zspec), (yi, yspec), (xi, xspec) = axis_wise_slices
    assert zi == 0 and yi == 1 and xi == 2
    tilespecs = []
    for zs, ys, xs in product(zspec, yspec, xspec):
        tilespecs.append(tuple((zs, ys, xs)))
    
    return tuple(tilespecs)



def add_offsets(
    offsets: tuple[int, int, int],
    tilespecs: tuple[Tilespec3D, ...]
) -> tuple[Tilespec3D, ...]:
    """
    Add offsets tilespecs.
    Useful if tilespecs are computed for an inner volume and need to be
    adjusted to be correctly located in the outer volume.

    Parameters
    ----------
    offsets : tuple[int, int, int]
        Offsets to add to the tilespecs.
        Offsets are given in the order (z, y, x).

    tilespecs : tuple[Tilespec3D, ...]
        Tilespecs to add offsets to.
        Tilespecs are given as tuples of slices in the order (z, y, x).

    Returns
    -------
    tuple[Tilespec3D, ...]
        Tilespecs with offsets added.
    """
    modified_tilespecs = []
    for tilespec in tilespecs:
        tilespec_offsetted = tuple(
            slice(slc.start + delta, slc.stop + delta) for slc, delta in zip(tilespec, offsets)
        )
        modified_tilespecs.append(tilespec_offsetted)
    return tuple(modified_tilespecs)



def compute_tiles_cuboidal(
    volshape: tuple[int, int, int],
    tileshape: tuple[int, int, int],
    rel_standoff: RelStandoff | None = None,
    abs_standoff: AbsStandoff | None = None
) -> tuple[Tilespec3D, ...]:
    """
    Compute the tilespecs for tiles of the given shape filling the
    cuboidal volume with the given `volshape`.
    Via `rel_standoff` and `abs_standoff` the user can specify
    a safety margin for the volume to be tiled.
    The standoff can be specified as fractions of the axis size
    (relative) or as absolute values (absolute) in the format (z, y, x).

    Parameters
    ----------
    volshape : tuple[int, int, int]
        Shape of the volume to be tiled.
        The order is (z, y, x).

    tileshape : tuple[int, int, int]
        Shape of the tile to be used.
        The order is (z, y, x).

    rel_standoff : tuple[float | None, float | None, float | None] | None, optional
        Relative standoff safety margin. Tiling will be done
        with respect to the inner volume.
        If None, no standoff will be applied.
        If a value is None, the corresponding axis will not have a standoff.
        Default is None.

    abs_standoff : tuple[int | None, int | None, int | None] | None, optional
        Absolute standoff safety margin. Tiling will be done
        with respect to the inner volume.
        If None, no standoff will be applied.
        If a value is None, the corresponding axis will not have a standoff.
        Default is None.
    
    Returns
    -------
    tuple[Tilespec3D, ...]
        Tilespecs for the tiles filling the volume.
    """
    D, H, W = volshape
    
    if abs_standoff and rel_standoff:
        raise ValueError('cannot simulatenously specify absolute and relative standoff - use only one')
        
    elif abs_standoff:
        # filter to move Nones (user wants to standoff for this axis) to zeros
        abs_standoff = tuple(s if s is not None else 0 for s in abs_standoff)
        delta_D, delta_H, delta_W = tuple(f for f in abs_standoff)
        
    elif rel_standoff:
        # filter to move Nones (user wants to standoff for this axis) to ones
        rel_standoff = tuple(s if s is not None else 0.0 for s in rel_standoff)
        delta_D, delta_H, delta_W = tuple(
            (int(np.rint(s * f)) for s, f in zip(volshape, rel_standoff))
        )
    else:
        delta_D, delta_H, delta_W = (0, 0, 0)
    
    if abs_standoff or rel_standoff:
        if abs_standoff:
            # filter to move Nones (user wants to standoff for this axis) to zeros
            abs_standoff = tuple(s if s is not None else 0 for s in abs_standoff)
            delta_D, delta_H, delta_W = tuple(f for f in abs_standoff)
        elif rel_standoff:
            # filter to move Nones (user wants to standoff for this axis) to ones
            rel_standoff = tuple(s if s is not None else 0.0 for s in rel_standoff)
            delta_D, delta_H, delta_W = tuple(
                (int(np.rint(s * f)) for s, f in zip(volshape, rel_standoff))
            )
        D_inner = D - delta_D
        H_inner = H - delta_H
        W_inner = W - delta_W
        shape_inner = (D_inner, H_inner, W_inner)
        tiles = _compute_tiles_cuboidal(shape_inner, tileshape)
        tiles = add_offsets(offsets=(delta_D, delta_H, delta_W), tilespecs=tiles)
    
    else:
        tiles = _compute_tiles_cuboidal(volshape, tileshape)
        
    return tiles



class CuboidalVolumeTileBuilder:
    """
    Build tilespecs for a standard cuboidal volume.
    The maximally possible volume is tiled with tiles of the given shape.

    Parameters
    ----------
    baseshape : tuple[int, int, int]
        Shape of the volume to be tiled.

    tileshape : tuple[int, int, int]
        Shape of the tile to be used.

    relative_standoff : tuple[float | None, float | None, float | None] | None, optional
        Relative standoff safety margin. Tiling will be done
        with respect to the inner volume.
        If None, no standoff will be applied.
        If a value is None, the corresponding axis will not have a standoff.
        Default is None.

    absolute_standoff : tuple[int | None, int | None, int | None] | None, optional
        Absolute standoff safety margin. Tiling will be done
        with respect to the inner volume.
        If None, no standoff will be applied.
        If a value is None, the corresponding axis will not have a standoff.
        Default is None.

    prepend_wildcards : int, optional
        Prepend wildcard (i.e. full selecting slice objects) for any
        generalized dimensions such as channel or batch.
        Defaults to 1.

    Attributes
    ----------
    baseshape : tuple[int, int, int]
        Shape of the volume to be tiled.
    
    tileshape : tuple[int, int, int]
        Shape of the tile to be used.

    tiles : tuple[Tilespec3D, ...]
        Tilespecs for the tiles filling the volume.
        Tilespecs are given as tuples of slices in the order (z, y, x).
    """
    def __init__(
        self,
        baseshape: tuple[int, int, int],
        tileshape: Tileshape3D,
        relative_standoff: RelStandoff | None = None,
        absolute_standoff: AbsStandoff | None = None,
        prepend_wildcards: int = 1
    ) -> None:

        self.baseshape = baseshape
        self.tileshape = tileshape
        self._absolute_standoff = absolute_standoff
        self._relative_standoff = relative_standoff
        # first build 3D tilespecs and then prepend wildcards
        tiles = compute_tiles_cuboidal(
            baseshape,
            tileshape,
            rel_standoff=relative_standoff,
            abs_standoff=absolute_standoff
        )
        self._tiles = self._prepend_wildcards(
            tilespecs=tiles,
            prepend_wildcards=prepend_wildcards
        )

    @property
    def tiles(self) -> tuple[Tilespec3D, ...]:
        """
        Tilespecs for the tiles filling the volume.
        """
        return self._tiles
    
    @staticmethod
    def _prepend_wildcards(
        tilespecs: Sequence[Tilespec3D],
        prepend_wildcards: int) -> tuple[TilespecND, ...]:
        """
        Prepend wildcard (i.e. full selecting slice objects) for any
        generalized dimensions such as channel or batch.
        """
        wildcards = tuple(np.s_[:] for _ in range(prepend_wildcards))
        # expand every 2D slice tuple into the third dimension along z axis
        tiles = []
        for tile in tilespecs:
            tiles.append(
                tuple((*wildcards, *tile))
            )
        return tuple(tiles)