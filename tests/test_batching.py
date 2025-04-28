import numpy as np
import pytest

from woodnet.datasets.batching import choice_and_choicemask, generate_batch_indices

@pytest.mark.parametrize(
    'elements,size,seed',
    [
        (np.arange(10), 5, 42),
        (np.arange(25), 5, None),
        (np.arange(75), 1, 42),
        (np.arange(333), 1, None),
        (np.arange(371), 10, 42),
        (np.arange(10), 7, None),
        (np.arange(1000), 1000, None)
    ]
)
def test_choice_and_choicemask_produces_return_values_of_correct_length(elements, size, seed):
    """
    Test that the function returns the correct number of elements and mask.
    """
    choice, mask = choice_and_choicemask(elements=elements, size=size, seed=seed)
    assert len(choice) == size
    assert len(mask) == len(elements)



@pytest.mark.parametrize(
    'n_elements,batch_size,seed',
    [
        (10, 11, 42),
        (10, 20, 42),
        (20, 21, 123),
        (15, 150, 0),
        (100, 101, None),
    ]
)
def test_fails_on_batchsize_larger_than_elements(n_elements, batch_size, seed):
    """
    Test that the function fails when batch size is larger than number of elements.
    """
    with pytest.raises(ValueError):
        generate_batch_indices(n_elements=n_elements, batch_size=batch_size, seed=seed)


@pytest.mark.parametrize(
    'n_elements,batch_size,seed',
    [
        (10, 1, 42),
        (10, 2, 42),
        (20, 3, 123),
        (15, 4, 0),
        (100, 5, None),
        (100, 7, 1701),
        (1000, 256, 42)
    ]
)
def test_all_elements_chosen_at_least_once(n_elements, batch_size, seed):
    expected_elements = set(range(n_elements))
    batch_indices = generate_batch_indices(
        n_elements=n_elements,
        batch_size=batch_size,
        seed=seed,
        drop_last=False
    )
    index_set = set(batch_indices.flat)
    # `set`` - `other``: computes the set of elements that are in `set` but not in `other`
    # in batch indices, there may be an additional `-1` element
    assert index_set - expected_elements == {-1} or index_set - expected_elements == set()
    # we should not have indices that are not in the batch indices
    assert expected_elements - index_set == set()


@pytest.mark.parametrize(
    'n_elements,batch_size,seed',
    [
        (10, 1, 42),
        (15, 4, 0),
        (100, 5, None),
        (100, 7, 1701),
        (1000, 256, 42),
        (1000, 735, None)
    ]
)
def test_batch_size_is_homogenous_for_drop_last(n_elements, batch_size, seed):
    batch_indices = generate_batch_indices(
        n_elements=n_elements,
        batch_size=batch_size,
        seed=seed,
        drop_last=True
    )
    # get indices per batch
    batch_sets = [set(batch) for batch in batch_indices]
    for batch_set in batch_sets:
        assert len(batch_set) == batch_size



@pytest.mark.parametrize(
    'n_elements,batch_size,seed',
    [
        (10, 1, 42),
        (10, 2, 42),
        (20, 3, 123),
        (15, 4, 0),
        (100, 5, None),
        (100, 7, 1701),
        (1000, 256, 42)
    ]
)
def test_all_elements_chosen_at_max_once(n_elements, batch_size, seed):
    expected_elements = set(range(n_elements))
    batch_indices = generate_batch_indices(
        n_elements=n_elements,
        batch_size=batch_size,
        seed=seed,
        drop_last=False
    )
    # get indices per batch
    *batch_sets, last = [set(batch) for batch in batch_indices]
    seen = set()
    for batch_set in batch_sets:
        assert len(batch_set) == batch_size
        assert seen.isdisjoint(batch_set)
        seen = seen | batch_set

    # the last batch may be smaller than `batch_size` and can be padded with `-1`
    last_set = set(last)
    if n_elements % batch_size == 0:
        assert len(last_set) == batch_size
    else:
        # this throws if `-1` is not present which should be the case due to padding
        _ = last_set.remove(-1)

    assert seen.isdisjoint(last_set)
    assert seen | last_set == expected_elements
