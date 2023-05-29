import numpy as np

from typing import Iterable, Optional
from math import sqrt

def mask_inside_circle(i, j, radius, shape):
    i_off = shape[0] // 2
    j_off = shape[1] // 2
    return np.sqrt((i - i_off)**2 + (j - j_off)**2) < radius


def secant_length(point, radius):
    x, y = point
    diameter = 2 * radius
    return diameter - np.array([2*x, 2*y])


def f(x, radius):
    return -np.sqrt(radius**2 - (x - radius)**2) + radius


def initial_point(a, radius, n_row_elements):
    x_0 = radius - np.sqrt(radius**2 - ((a * n_row_elements) / 2)**2)
    y_0 = f(x_0, radius)
    return np.array((x_0, y_0))


def is_square(shape: Iterable[int],
              dims: Optional[tuple[int]] = None) -> bool:
    """
    Check if shape-like object is square along the indicate dimensions.
    Defaults to None, meaning that all dimensons are considered.    
    """
    if dims is None:
        dims = np.s_[:]
    else:
        dims = np.array(dims)
    sizes = np.array(shape)[dims]
    a = sizes[0]
    if all(a == s for s in sizes[1:]):
        return True
    return False


def is_3D(shape: tuple[int]) -> bool:
    return len(shape) == 3


def retrieve_pattern(a: int, radius: int, reltol: float = 0.01) -> str:
    """
    Retrieve a ssymmetric positioning pattern for the squares inside the circle.
    
    Parameters
    ----------
    
    a : int
        Edge length of the squares inside the circle.
        
    radius : int
        Radius of the circle and simultaneously the edge length
        of the bounding box.
        
    reltol : float, optional
        Relaxation of fitting paramter bz this relative value/
        Defaults to `0.01`.
    """
    ratio = radius / a
    lower, upper = ratio * (1 - reltol), ratio * (1 + reltol)
    pattern_mapping = {
        sqrt(2) / 2 : '1',
        sqrt(5) / 2 : '2',
        5 * sqrt(17) / 16 : '1;2',
        sqrt(2) : '2;2',
        sqrt(10) / 2 : '1;3;1',
        sqrt(13) / 2 : '2;3;2',
        3 * sqrt(3) / 2 : '3;4;3',
        sqrt(5) : '2;4;4;2',
        5 / 2 : '3;4;4;3',
        sqrt(34) / 2 : '3;5;5;5;3',
        sqrt(41) / 2 : '2;5;6;6;5;2',
        5 * sqrt(2) / 2 : '1;5;6;7;6;5;1'
    }
    *_, (factor, pattern) = [
        (factor, pattern) for factor, pattern in pattern_mapping.items()
        if factor < upper
    ]
    return pattern
    

def compute_vertical_increment(point, a, row_count, radius):
    """
    Compute vertical increment for the starting point on the circle.
    
    Parameters
    ----------
    
    point : tuple, list, np.ndarray
        The tarting point in (x, y) coordinates
        
    a : int
        Sqaure edge length.
        
    row_count : int
        Number of rows inside the circle.
        
    radius : int
        Radius of the circle.
    """
    # incrementation is irrelevant for this case and we avert the singularity
    # inside the inter-qaure spaceing computation
    if row_count == 1:
        return 0
    x, y = point
    vs, hs = secant_length(point, radius)
    # compute the inter-square distance
    inter = (vs - row_count * a) / (row_count - 1)
    delta_x = a + inter
    return delta_x


def compute_horizontal_increment(point, a, column_count, radius):
    """Compute the horizontal increment from the starting point."""
    # incrementation is irrelevant for this case and we avert the singularity
    # inside the inter-qaure spaceing computation
    if column_count == 1:
        return 0
    x, y = point
    vs, hs = secant_length(point, radius)
    # compute inter-square distance
    inter = (hs - column_count * a) / (column_count - 1)
    delta_y = a + inter
    return delta_y


def compute_vertex_coordinates(pattern: str, a: int, radius: int) -> list[list[float]]:
    """
    Compute the vertex cooridanates for the given pattern.
    
    Parameters
    ----------
    
    pattern : str
        Row and columen pattern of the squares inside the
        circle.
        
    a : int
        Square edee lenth in voxel units.
        
    radius : int
        Radius of the circle.
        2 * radius is also the edge length of the
        bounding box.
    """
    pattern = [int(item) for item in pattern.split(';')]
    rowcount = len(pattern)
    
    init_point = initial_point(a, radius, pattern[0])
    x0, y0 = init_point
    delta_x = compute_vertical_increment(init_point, a, rowcount, radius)
   
    vertex_points = []
    
    for row_idx, n_col in enumerate(pattern):        
        # new row starting point on circle edge
        row_x0 = x0 + row_idx * delta_x
        # increment for square node coordinates
        a_x = a
        # if the squares center point is in the lower half space, we
        # switch to the lower left node of the square to represent the position
        if (row_x0 + a / 2) >= radius:
            row_x0 += a
            a_x = -a
        
        row_y0 = f(row_x0, radius)
        row_point = np.array([row_x0, row_y0])
        delta_y = compute_horizontal_increment(row_point, a, n_col, radius)
                
        for col_idx in range(n_col):
            vertex_points.append(
                [
                    [row_x0, row_y0 + col_idx * delta_y],
                    [row_x0 + a_x, row_y0 + col_idx * delta_y],
                    [row_x0, row_y0 + col_idx * delta_y + a],
                    [row_x0 + a_x, row_y0 + col_idx * delta_y + a]
                ]
            )
    return vertex_points


def compute_tile(vertex_points: Iterable[Iterable[float]]) -> tuple[slice]:
    """
    Compute slice objects that encode the tile from vertex coordinate
    points of squares.
    """
    vertex_points = np.array(vertex_points)
    print(vertex_points)
    print(vertex_points.shape)
    
    mins = np.rint(np.min(vertex_points, axis=0))
    maxs = np.rint(np.max(vertex_points, axis=0))
    slices = []
    for min_, max_ in  zip(mins, maxs):
        slices.append(slice(int(min_), int(max_)))
    return tuple(slices)
    
    
def compute_tiles(*tile_vertex_points: Iterable[Iterable[float]]) -> tuple[tuple[slice]]:
    slices = []
    for vertex_points in tile_vertex_points:
        slices.append(compute_tile(vertex_points))
    return tuple(slices)
    

def compute_z_increment(zsize: int, a: int, layers: int) -> int:
    """
    Compute increment of z voxels between successive layer tops.
    
    Parameters
    ----------
    
    zsize : int
        Total size/extent along the z-axis in voxels.
        
    a : int
        Edge length of the tile in voxels.
        
    layers : int
        Number of layers along the z axis.
    """
    d = int(np.floor((zsize - layers * a) / (layers - 1)))
    return a + d


def compute_z_layer_count(zsize: int, a: int) -> int:
    """
    Compute number of layers along the z axis for a tile with edge length a.
    
    Parameters
    ---------
    
    zsize : int^b
        Total extent of the volume along the z axis.
    
    a : int
        Edge length of the tile in voxels.
    """
    return int(np.floor(zsize / a))


def compute_z_slices(a: int, z_increment: int, layers: int) -> list[slice]:
    """
    Compute the list of slice objects that partition the z-dimension
    into the given number of layers.
    """
    slices = []
    for i in range(layers):
        slices.append(
            slice(i*z_increment, i*z_increment + a)
        )
    return slices


class TileBuilder:
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
        must be close to bondary within radius_atol.
    """
    radius_atol: int = 10
    packing_reltol: float = 0.01
    
    def __init__(self, baseshape: tuple[int], tileshape: tuple[int], radius: int,
                 prepend_wildcards: int = 1):
        
        for shape, dims in zip((baseshape, tileshape), ((1, 2), None)):
            if not is_3D(shape):
                raise ValueError(f'expected 3D shape, got  ndim = {len(shape)}')
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
        self.zsize = baseshape[-1]
        self.layers = compute_z_layer_count(self.zsize, self.a)
        self.prepend_wildcards = prepend_wildcards
    
    
    @property
    def tiles(self) -> list[slice]:
        pattern = retrieve_pattern(self.a, self.radius, self.packing_reltol)
        vertex_coordinates = compute_vertex_coordinates(pattern, self.a, self.radius)
        tile_slices = compute_tiles(*vertex_coordinates)
        z_increment = compute_z_increment(self.zsize, self.a, self.layers)
        z_slices = compute_z_slices(self.a, z_increment, self.layers)
        # wildcards may select any frontal channel or batch dimensions
        wildcards = tuple(np.s_[:] for _ in range(self.prepend_wildcards))
        # expand every 2D slice tuple into the third dimension along z axis
        tiles = []
        for z_slice in z_slices:
            for tile in tile_slices:
                tiles.append(
                    tuple((*wildcards, z_slice, *tile))
                )
        return tiles
        