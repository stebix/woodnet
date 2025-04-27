from pathlib import Path

import torch

from woodnet.datasets.planar.slicebased import build_from_zarr
from woodnet.transformations.transformer import Transformer



def test_smoke():
    path = Path('/home/jannik/storage/wood/phase/joint4/CT20.zarr')
    internal_path = '/resampled/pure/sf-0_1'
    phase = 'train'
    transformer = Transformer.from_configurations([])
    scaling_policy = 'default'
    classlabel_mapping = {
        'acer' : 0,
        'pinus' : 1
    }
    dataset = build_from_zarr(
        path=path,
        internal_path=internal_path,
        phase=phase,
        transformer=transformer,
        scaling_policy=scaling_policy,
        classlabel_mapping=classlabel_mapping
    )

    data, label = dataset[0]
    assert isinstance(data, torch.Tensor)
    assert isinstance(label, torch.Tensor)  

    print(data.shape)
    print(label.shape)