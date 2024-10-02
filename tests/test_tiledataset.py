import torch

from woodnet.datasets.volumetric import TileDataset, TileDatasetBuilder
from woodnet.transformations.transformer import Transformer
from woodnet.transformations import from_configurations


def test_smoke_builder():
    phase = 'train'
    tileshape = (256, 256, 256)
    configs = [
        {'name' : 'Normalize', 'mean' : 110, 'std' : 950}
    ]
    IDs = ['CT10', 'CT20']
    builder = TileDatasetBuilder()
    raise Exception

    datasets = builder.build(IDs,
                             phase=phase, tileshape=tileshape,
                             transform_configurations=configs)    

    assert len(datasets) == 2

    # CT10 should be acer, CT20 should be pinus
    ds_acer, ds_pinus = datasets
    item = next(iter(ds_acer))

    # deeply inspect structures of returned data
    assert isinstance(item, tuple)
    subvolume, label = item
    assert label.item() == 0, 'acer should be class 0'