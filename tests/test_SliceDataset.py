import torch

from woodnet.dataobjects import CachingSlice
from woodnet.datasets.planar import SliceDataset


def test_basic_initialization_at_train_phase(tiff_generator, fingerprint):
    """
    Basic smoke test whether the dataset object works at all.
    """
    N = 10
    classlabel_mapping = {'ahorn' : 0, 'kiefer' : 1}
    filepaths = tiff_generator.make_many(n=N)
    slices = []
    for index, fp in enumerate(filepaths):
        slices.append(
                CachingSlice(filepath=fp,
                             fingerprint=fingerprint,
                             index=index)
        )
    dataset = SliceDataset(
        phase='train', slices=slices,
        classlabel_mapping=classlabel_mapping,
        transformer=None
    )
    assert len(dataset) == N
    # check data consistency
    index = 5
    expected_data = slices[index].data
    expected_classlabel = slices[index].class_
    expected_label = classlabel_mapping[expected_classlabel]
    (data, label) = dataset[index]
    assert torch.allclose(data, torch.tensor(expected_data))
    assert torch.allclose(label, torch.tensor(expected_label))
    
    


