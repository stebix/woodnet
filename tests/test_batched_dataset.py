from pathlib import Path

import numpy as np
import pytest
import zarr

from woodnet.datasets.planar.batched import BatchedTileDataset
from woodnet.datasets.fingerprints import Fingerprint, StatParams
from woodnet.datasets.planar.slicebased import CentroidTileSubselector
from woodnet.transformations.transformer import Transformer

from woodnet.datasets.batching import are_row_permutations

@pytest.mark.parametrize(
    'batch_size',
    [1, 2, 3, 25, 64]
)
def test_smoke_initialization(batch_size):
    fpath = Path('/home/jannik/storage/wood/phase/joint4/CT10.zarr')
    internal_path = 'resampled/pure/sf-0_17'
    fprint = Fingerprint.from_zarr(fpath, internal_path)
    stats = StatParams.from_zarr(fpath, internal_path)
    zarrobj = zarr.convenience.open(fpath, mode='r')

    subselector = CentroidTileSubselector(z_spacing=2)

    data = zarrobj[internal_path][...]
    data_filtered = subselector(data)

    data_filtered_zsize = data_filtered.shape[1]

    transformer = Transformer()
    
    dataset = BatchedTileDataset(
        phase='train',
        data=data_filtered,
        fingerprint=fprint,
        stats=stats,
        transformer=transformer,
        classlabel_mapping={'acer': 0, 'pinus': 1},
        batch_size=batch_size,
        drop_last=True,
        seed=42
    )

    assert len(dataset) == data_filtered_zsize // dataset.batch_size

    item, label = dataset[1]
    assert item.shape[0] == dataset.batch_size
    # expecting layout (N x C x H x W)
    assert item.shape[-1] == data_filtered.shape[-1], 'dataset item W dimension mismatch'
    assert item.shape[-2] == data_filtered.shape[-2], 'dataset item H dimension mismatch'
    assert label.shape[0] == dataset.batch_size



@pytest.mark.parametrize(
    'batch_size',
    [1, 2, 3, 25, 32]
)
def test_shuffle_batch_indices_actually_produces_different_batch_order(batch_size):
    fpath = Path('/home/jannik/storage/wood/phase/joint4/CT10.zarr')
    internal_path = 'resampled/pure/sf-0_17'
    fprint = Fingerprint.from_zarr(fpath, internal_path)
    stats = StatParams.from_zarr(fpath, internal_path)
    zarrobj = zarr.convenience.open(fpath, mode='r')

    subselector = CentroidTileSubselector(z_spacing=2)

    data = zarrobj[internal_path][...]
    data_filtered = subselector(data)

    data_filtered_zsize = data_filtered.shape[1]

    transformer = Transformer()
    
    dataset = BatchedTileDataset(
        phase='train',
        data=data_filtered,
        fingerprint=fprint,
        stats=stats,
        transformer=transformer,
        classlabel_mapping={'acer': 0, 'pinus': 1},
        batch_size=batch_size,
        drop_last=True,
        seed=42
    )
    print(len(dataset))
    assert len(dataset) == data_filtered_zsize // dataset.batch_size

    for _ in range(3):
        previous_batch_indices = dataset.batch_indices.copy()
        dataset.shuffle_batch_indices()
        current_batch_indices = dataset.batch_indices.copy()
        # NOTE: this only holds if the dataset has more than two batches
        assert not np.allclose(previous_batch_indices, current_batch_indices), f'dset length: {len(dataset)}'
        assert are_row_permutations(previous_batch_indices, current_batch_indices)