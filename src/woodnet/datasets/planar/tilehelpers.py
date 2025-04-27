import numpy as np


def compute_centroid_square(
    shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """
    Compute the coordinates of a square that is centered in the data
    with a circular ROI.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the data. The second-to-last dimension
        is assumed to be the height and the last dimension
        is assumed to be the width.
        Any number of preceding dimensions are allowed.

    Returns
    -------
    tuple[int, int, int, int]
        Coordinates of the square in the format (y, x, height, width).
        The height and width are equal (square region).
    """
    *_, H, W = shape
    diameter = min(H, W)
    a = diameter / np.sqrt(2)
    center_x = W // 2
    center_y = H // 2
    sx = int(np.rint(center_x - a / 2))
    sy = int(np.rint(center_y - a / 2))
    a = int(np.rint(a))
    return (sy, sx, a, a)


def to_slices(sy: int, sx: int, dy: int, dx: int) -> tuple[slice, slice]:
    """
    Convert a specification of a rectangle into slices.

    Parameters
    ----------
    sy : int
        Starting y-coordinate of the rectangle.

    sx : int
        Starting x-coordinate of the rectangle.

    dy : int
        Size of the rectangle in the y-direction.

    dx : int
        Size of the rectangle in the x-direction.

    Returns
    -------
    tuple[slice, slice]
        Slices for the y and x dimensions.
        The slices are in the format (y_slice, x_slice).
    """
    yslice = slice(sy, sy+dy)
    xslice = slice(sx, sx+dx)
    return (yslice, xslice)