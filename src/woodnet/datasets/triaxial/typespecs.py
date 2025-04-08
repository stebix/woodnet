"""
Commonly used specifications and ype aliases for triaxial data.

@Author: Jannik Stebani 2025
"""
from typing import NamedTuple, TypeAlias, Union

Tilespec3D: TypeAlias = tuple[slice, slice, slice]
TilespecND: TypeAlias = tuple[slice, ...]
Tileshape3D: TypeAlias = tuple[int, int, int]

Planespec3D: TypeAlias = Union[
    tuple[int, slice, slice],
    tuple[slice, int, slice],
    tuple[slice, slice, int]
]

TriaxPlanes: TypeAlias = tuple[Planespec3D, Planespec3D, Planespec3D]


class TriaxPlanesSpec(NamedTuple):
    """Specification of the three orthogonal planes"""
    zplane: Planespec3D
    yplane: Planespec3D
    xplane: Planespec3D


class TilePlanespec3D(NamedTuple):
    """
    Jointly specify the tile index and the planes to extract from the tile.
    """
    tidx: int
    triax_planespec: TriaxPlanesSpec

