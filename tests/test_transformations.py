import torch

import pytest

from woodnet.transformations.transforms import Identity
from woodnet.transformations.buildtools import from_configurations


def test_transformer_from_empty_configurations_is_identity():
    configurations = []
    transformer = from_configurations(configurations)
    x = torch.randn((1, 16, 16, 16))
    out = transformer(x)
    assert torch.allclose(x, out)