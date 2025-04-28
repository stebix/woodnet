from collections import Counter

import numpy as np

PAD_VALUE: int = -1

def choice_and_choicemask(
    elements: np.ndarray,
    size: int,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random selection of `size` entries from `elements` without replacement.
    In addition to the pure numpy function, we also get the boolean mask of the
    choice into the choice value reservoir `elements` with *True* at positions
    that chosen and *False* at positions that were not chosen.
    """
    assert elements.ndim == 1, f'expecting 1D array, but got {elements.ndim}D'
    rng = np.random.default_rng(seed)
    n = len(elements)
    possible_indices = np.arange(0, n)
    idx_choice = rng.choice(possible_indices, size=size, replace=False)
    choice = elements[idx_choice]
    mask = np.full(shape=len(elements), fill_value=True)
    mask[idx_choice] = False
    return (choice, mask)
    

def generate_batch_indices(
    n_elements: int,
    batch_size: int,
    seed: int,
    drop_last: bool = False,
) -> np.ndarray:
    """
    Generate indices for batches of size `batch_size` from a total of
    `n_elements` elements. The indices are generated randomly and
    without replacement.
    
    The last batch index array may be padded with `-1` up to `batch_size`
    if `drop_last` is set to False.
    
    If `drop_last` is set to True, the
    last batch will be dropped if it is smaller than `batch_size`.
    """
    if n_elements < batch_size:
        raise ValueError(
            f'batch size {batch_size} is larger than number of elements {n_elements}'
        )
    elements = np.arange(0, n_elements)
    indices = []
    while len(elements) >= batch_size:
        choice, mask = choice_and_choicemask(elements, size=batch_size, seed=seed)
        indices.append(choice)
        # mask chosen indices to exclude for next random selection
        elements = elements[mask]
    
    if drop_last or len(elements) == 0:
        return np.array(indices)
    
    indices.append(
        np.concatenate(
            (elements, np.full(fill_value=PAD_VALUE, shape=(batch_size - len(elements)))))
    )
    return np.array(indices)


def filter_pad_value(
    indices: np.ndarray,
    pad_value: int = PAD_VALUE
) -> np.ndarray:
    """
    Convenient removal of the padding value from a 1D batch indices array.
    """
    if not indices.ndim == 1:
        raise ValueError(
            f'indices should be 1D, but got {indices.ndim}D'
        )
    return indices[indices != pad_value]



def are_row_permutations(
        array1: np.ndarray,
        array2: np.ndarray
) -> bool:
    """
    Check if two 2D Numpy arrays have the same rows, possibly in a different order.
    
    Parameter
    ----------
    array1 : numpy.ndarray
        First 2D array of integers
    
    array2 : numpy.ndarray
        Second 2D array of integers
        
    Returns
    -------
    bool
        True if one array is a row permutation of the other, False otherwise
        
    Raises
    ------
    TypeError
        If either array is not of integer data type
    """
    if not np.issubdtype(array1.dtype, np.integer):
        raise TypeError('First array must contain integer data type, got {array1.dtype}')
    if not np.issubdtype(array2.dtype, np.integer):
        raise TypeError('second array must contain integer data type, got {array2.dtype}')
    
    if array1.shape != array2.shape:
        return False
    
    # Convert rows to tuples so they can be hashable for comparison
    rows1 = [tuple(row) for row in array1]
    rows2 = [tuple(row) for row in array2]
    
    # Count occurrences of each row
    counter1 = Counter(rows1)
    counter2 = Counter(rows2)
    return counter1 == counter2