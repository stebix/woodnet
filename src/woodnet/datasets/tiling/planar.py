"""
Tiling for planar (i.e. 2D) patches of 3D volumes.

@Author: Jannik Stebani 2025
"""
import numpy as np

from woodnet.datasets.tiling.utils import is_square, is_2D
from woodnet.datasets.tiling.cylindrical import compute_vertex_coordinates, compute_tiles, retrieve_pattern

class PlanarTileBuilder:
    """
    Compute tiles (i.e. cubic subvolumes) for a cylindrical region inside a 3D voxel volume.
    Individual tiles are stored as 3-tuples of slice objects with (z, {x, y}) axis ordering.
    
    Parameters
    ----------
    
    baseshape: tuple of int
        Shape of the basal embedding voxel volume. 
        The volume must be square for the {x, y} dimensions,
        e.g. (950, 1200, 1200).
        
    tileshape : tuple of int
        Shape of a square tile subvolume, e.g. (256, 256, 256)
        
    radius : int
        Radius of the embedded cylinder. The cylinder is expected
        to fill the embedding volume almost fully radius-wise, i.e.
        must be close to boundary within radius_atol.

    prepend_wildcards: int, optional
        Prepend wildcard (i.e. full selecting slice objects) for any
        generalized dimensions such as channel or batch.
        Defaults to 1.
    """
    radius_atol: int = 10
    packing_reltol: float = 0.01
    
    def __init__(self,
                 baseshape: tuple[int],
                 tileshape: tuple[int],
                 radius: int,
                 prepend_wildcards: int = 0):
        
        for shape, dims in zip((baseshape, tileshape), (None, None)):
            if not is_2D(shape):
                raise ValueError(f'expected 2D shape, got ndim = {len(shape)}')
            if not is_square(shape, dims=dims):
                raise ValueError(f'expected square shape but got {shape}')
        # check for fitting of cylinder to embedding volume 
        if (2*radius - baseshape[-1]) > self.radius_atol:
            raise ValueError(f'Embedded circle radius {radius} and baseshape {baseshape} '
                             f'exceed maximum tolerance of {self.radius_atol}')
        
        self.baseshape = baseshape
        self.tileshape = tileshape
        self.radius = radius
        self.a = tileshape[0]
        # see there big bug due to magic numbers in the frickin code
        # FUUUUUUUUUUUUUUUUUUUUUUUUUUUUU *ragemode*
        self.prepend_wildcards = prepend_wildcards
    
    
    @property
    def tiles(self) -> list[slice]:
        pattern = retrieve_pattern(self.a, self.radius, self.packing_reltol)
        vertex_coordinates = compute_vertex_coordinates(pattern, self.a, self.radius)
        tile_slices = compute_tiles(*vertex_coordinates)
        # wildcards may select any leading channel or batch dimensions
        wildcards = tuple(np.s_[:] for _ in range(self.prepend_wildcards))
        # expand every 2D slice tuple into the third dimension along z axis
        tiles = []
        for tile in tile_slices:
            tiles.append(
                tuple((*wildcards, *tile))
            )
        return tiles
    
        