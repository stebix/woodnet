"""
Tooling to broadcast the application of a function to a sequence of arrays.
"""
from typing import TypeAlias, Union
from collections.abc import Callable, Mapping, Hashable

import numpy as np


ArraySequence: TypeAlias = Union[
    list[np.ndarray],
    tuple[np.ndarray],
    Mapping[Hashable, np.ndarray]
]


def map_func_to_arrays(
    array_sequence: ArraySequence | np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
) -> ArraySequence | np.ndarray:
    """
    Apply a function to each element of a sequence of arrays.
    If the input is a single array, the output will be a single array.
    The function preserves the structure of the input sequence.
    Note: This function does not handle nested sequences.
    """
    if isinstance(array_sequence, np.ndarray):
        return func(array_sequence)
    elif isinstance(array_sequence, (list, tuple)):
        return type(array_sequence)(func(arr) for arr in array_sequence)
    elif isinstance(array_sequence, Mapping):
        return type(array_sequence)({k: func(v) for k, v in array_sequence.items()})
    else:
        raise TypeError(
            f'Expected a sequence of arrays or a single array, '
            f'but got {type(array_sequence)}.'
        )