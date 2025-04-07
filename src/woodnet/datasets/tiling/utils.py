from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np

def is_square(shape: Iterable[int],
              dims: Optional[tuple[int]] = None) -> bool:
    """
    Check if shape-like object is square along the indicate dimensions.
    Defaults to None, meaning that all dimensions are considered.    
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


def is_2D(shape: Sequence[int, int]) -> bool:
    return len(shape) == 2


def is_3D(shape: Sequence[int]) -> bool:
    return len(shape) == 3