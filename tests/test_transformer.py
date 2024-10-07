import torch
import pytest
from monai.transforms.intensity.array import GaussianSmooth

from woodnet.transformations.transformer import Transformer
from woodnet.inference.parametrized_transforms import (generate_parametrized_transforms,
                                                       ParametrizedTransform)

@pytest.fixture
def smoothing_parametrized_transform():
    specification = {
        'name' : 'GaussianSmoothie',
        'class_name' : 'GaussianSmooth',
        'parameters' : [
            {'sigma' : 1.0},
            {'sigma' : 2.0},
            {'sigma' : 3.0}
        ]
    }
    return generate_parametrized_transforms(specification)


def test_with_single_identity_transform():
    identity = lambda x: x
    transformer = Transformer(identity)
    x = torch.randn((1, 16, 16, 16))
    out = transformer(x)
    assert torch.allclose(x, out)


def test_with_single_oneify_transform():
    oneify = lambda x: x / x
    transformer = Transformer(oneify)
    x = torch.randn((1, 16, 16, 16))
    out = transformer(x)
    expected_out = torch.ones_like(out)
    assert torch.allclose(out, expected_out)


def test_with_parametrized_transform_in_init(smoothing_parametrized_transform):
    # build expected paramettrized transform manually for sanity check
    manual_smoother = GaussianSmooth(sigma=1.0)
    # this is the sigma = 1.0 parametrization of the GaussianSmoother
    smoother = smoothing_parametrized_transform[0]
    identity = lambda x: x / x
    transformer = Transformer(identity, parametrized_transform=smoother)
    x = torch.randn((1, 16, 16, 16))
    out = transformer(x)
    # The standard transform gets us an all-ones tensor on which the parametrized_transform
    # should be applied. This is a Gaussian smooth with sigma = 1.0 in this case.
    expected_out = manual_smoother(torch.ones_like(out))
    assert torch.allclose(out, expected_out)


def test_correct_setting_and_logging_of_parametrized_transform(smoothing_parametrized_transform):
    oneify = lambda x: x / x
    transformer = Transformer(oneify, oneify)
    for smoother in smoothing_parametrized_transform:
        transformer.parametrized_transform = smoother
        # build expected parametrized transform manually for sanity check
        sigma = smoother.parameters['sigma']
        manual_smoother = GaussianSmooth(sigma=sigma)
        x = torch.randn((1, 16, 16, 16))
        out = transformer(x)
        # The standard transform gets us an all-ones tensor on which the parametrized_transform
        # should be applied. This is a Gaussian smooth with variable sigma.
        expected_out = manual_smoother(torch.ones_like(out))
        assert torch.allclose(out, expected_out)
