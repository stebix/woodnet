import pytest
import torch
import itertools

from random import sample
from collections.abc import Sequence

from woodnet.inference.parametrized_transforms import (CongruentTransformList,
                                                       generate_parametrized_transforms,
                                                       ParametrizedTransform)
from woodnet.datasets.volumetric_inference import (TransformedTileDataset,
                                                   TransformedTileDatasetBuilder,
                                                   set_parametrized_transform)


def zeroize(inputs):
    return 0 * inputs

def make_ranges(lengths: Sequence[int]) -> list[range]:
    ranges = []
    start = 0
    for length in lengths:
        ranges.append(range(start, start + length))
        start = start + length
    return ranges

class Test_TransformedTileDatasetBuilder:
    @pytest.mark.slow
    def test_smoke_with_single_ID(self):
        builder = TransformedTileDatasetBuilder()
        datasets = builder.build(instances_ID=['CT10'], tileshape=(64, 64, 64))
        assert len(datasets) == 1
        dataset = datasets.pop()
        assert isinstance(dataset, TransformedTileDataset)


class Test_set_parametrized_transform:
    """
    Stylistic questions about test function names:

    - very long names that are as descriptive as possible but break the stylistic
      conventions very harshly

    - use more concise names and rely on the fact that the test functions are
      contained inside a module and inside a class that provide quite a lot
      of contextual information?
    """
    @pytest.fixture(scope='function')
    def single_transformed_dataset(self):
        builder = TransformedTileDatasetBuilder()
        datasets = builder.build(instances_ID=['CT10'], tileshape=(64, 64, 64), transform_configurations=None)
        assert len(datasets) == 1
        dataset = datasets.pop()
        return dataset

    @pytest.fixture(scope='function')
    def three_transformed_datasets(self):
        builder = TransformedTileDatasetBuilder()
        datasets = builder.build(instances_ID=['CT10', 'CT9', 'CT1'], tileshape=(64, 64, 64), transform_configurations=None)
        assert len(datasets) == 3
        return datasets
    
    @pytest.mark.slow
    def test_correct_setting_of_parametrized_transform_attribute_for_list_of_TransformedTileDataset(self, three_transformed_datasets):
        datasets = three_transformed_datasets
        ptf = ParametrizedTransform(name='Zeroizer', parameters={}, transform=zeroize)
        for subset in datasets:
            assert subset.parametrized_transform is None
        # meat of test here: this sets the parametrized transform
        set_parametrized_transform(datasets, transform=ptf)
        for subset in datasets:
            assert subset.parametrized_transform == ptf

    @pytest.mark.slow
    def test_correct_setting_of_parametrized_transform_attribute_for_ConcatDataset(self, three_transformed_datasets):
        # this frickin design choice where the datasets are instantiated directly from
        # large disk data >:()  reeeeeee stupido software architect
        separate_datasets = three_transformed_datasets
        ptf = ParametrizedTransform(name='Zeroizer', parameters={}, transform=zeroize)
        cct_dataset = torch.utils.data.ConcatDataset(separate_datasets)
        for subset in cct_dataset.datasets:
            assert subset.parametrized_transform is None
        # meat of test here: this sets the parametrized transform
        set_parametrized_transform(cct_dataset, transform=ptf)
        for subset in cct_dataset.datasets:
            assert subset.parametrized_transform == ptf

    @pytest.mark.slow
    def test_correct_setting_of_parametrized_transform_attribute_for_single_TransformedTileDataset(self, single_transformed_dataset):
        dataset = single_transformed_dataset
        ptf = ParametrizedTransform(name='Zeroizer', parameters={}, transform=zeroize)
        assert dataset.parametrized_transform is None
        # meat of test here: this sets the parametrized transform
        set_parametrized_transform(dataset, ptf)
        assert dataset.parametrized_transform == ptf

    @pytest.mark.slow
    def test_retrieved_data_is_actually_mutated_by_parametrized_transform(self, three_transformed_datasets):
        zero = torch.tensor(0.0)
        ptf = ParametrizedTransform(name='Zeroizer', parameters={}, transform=zeroize)
        cct_dataset = torch.utils.data.ConcatDataset(three_transformed_datasets)
        # setup for the concatenated datasets: we want three samples for every sub-dataset
        draws_per_subset: int = 3
        ranges = make_ranges([len(d) for d in three_transformed_datasets])
        sample_indices = itertools.chain.from_iterable(
            [sample(rng, k=draws_per_subset) for rng in ranges]
        )
        # actually set the parametrized transform
        set_parametrized_transform(cct_dataset, transform=ptf)
        for index in sample_indices:
            element = cct_dataset[index]
            data, _ = element
            assert torch.allclose(data, zero)


    