from typing import TypeAlias

import numpy as np

Tileshape3D: TypeAlias = tuple[int, int, int]
Tilespec3D: TypeAlias = tuple[slice, slice, slice]


def get_centered_tilespecs(
    shape: tuple[int, int, int],
    tileshape: Tileshape3D
) -> tuple[Tilespec3D, ...]:
    """
    Compute non-overlapping tile specification slices for a given volume shape.
    
    Generates tile specifications centered within the volume.
    
    Parameters
    ----------
    shape : tuple of int
        Shape tuple (D, H, W) representing the dimensions of the host volume.

    tileshape : Tileshape3D
        A tuple (dz, dy, dx) representing the dimensions of the desired tiles.
        
    Returns
    -------
    tuple of Tilespec3D
        A tuple of tile specifications. Each specification contains three slice 
        objects (slice_z, slice_y, slice_x) defining a tile.
        
    Raises
    ------
    ValueError
        If the tile size exceeds the volume size in any dimension or if inputs are invalid.
    
    Notes
    -----
    The function centers the entire grid of tiles within the volume, distributing
    any remaining space evenly around the edges.
    """
    D, H, W = shape
    dz, dy, dx = tileshape

    if not (dz > 0 and dy > 0 and dx > 0):
        raise ValueError('Tileshape3D elements must be positive')
    
    if not (D > 0 and H > 0 and W > 0):
        raise ValueError('volume shape must be positive')

    if dz > D or dy > H or dx > W:
        raise ValueError(
            f'requested Tileshape3D {dz, dy, dx} exceeds volume shape {D, H, W}'
        )

    # number of full tiles fit
    num_z = D // dz
    num_y = H // dy
    num_x = W // dx

    # occupied space by the tiles
    occupied_z = num_z * dz
    occupied_y = num_y * dy
    occupied_x = num_x * dx

    # Calculate centering offset for the *first* chunk
    offset_z = (D - occupied_z) // 2
    offset_y = (H - occupied_y) // 2
    offset_x = (W - occupied_x) // 2

    slices = []
    for i in range(num_z):
        start_z = offset_z + i * dz
        end_z = start_z + dz
        slice_z = slice(start_z, end_z)
        for j in range(num_y):
            start_y = offset_y + j * dy
            end_y = start_y + dy
            slice_y = slice(start_y, end_y)
            for k in range(num_x):
                start_x = offset_x + k * dx
                end_x = start_x + dx
                slice_x = slice(start_x, end_x)
                slices.append((slice_z, slice_y, slice_x))

    return tuple(slices)




class PreSelector:
    """
    Base class for pre-selectors.
    Pre-selectors are used to filter or select data before it is passed to the main processing pipeline.
    """
    def __call__(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement __call__ method'
        )



class PreTiler:
    def __self__(
        self,
        tileshape: Tileshape3D,
    ) -> None:
        """
        Pre-tiler for 3D data.
        """
        self.tileshape = tileshape


    def __call__(self, data: np.ndarray) -> list[np.ndarray]:
        """
        Pre-tile the data into smaller tiles of the specified shape.
        """
        *pre, D, H, W = data.shape
        wildcard = tuple(slice(None) for _ in range(len(pre)))
        tilespecs = get_centered_tilespecs(
            shape=(D, H, W),
            tileshape=self.tileshape
        )
        tiles = [
            data[(*wildcard, *tilespec)]
            for tilespec in tilespecs
        ]
        return tiles