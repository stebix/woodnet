"""
Test the model parameter extraction tooling utilized for
the histogram parameter logging.

@jsteb 2024
"""
import torch
import pytest

from woodnet.models.volumetric import ResNet3D
from woodnet.logtools.tensorboard.modelparameters.extraction import (extract_simple_resnet_parameters,
                                                                     convert_to_flat)


def test_manual_extraction():
    print(torch.__version__)
    net = ResNet3D(in_channels=1)

    net_fast = torch.compile(net)

    parameters, _ = extract_simple_resnet_parameters(net_fast)
    parameters = convert_to_flat(parameters)

    for key, value in parameters.items():
        print(f'{key} :: {type(value)} :: {value.shape}')