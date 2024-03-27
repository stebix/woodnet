"""
Test the model parameter extraction tooling utilized for
the histogram parameter logging.

@jsteb 2024
"""
import pytest

from woodnet.models.volumetric import ResNet3D
from woodnet.logtools.tensorboard.modelparameters.extraction import (extract_simple_resnet_parameters,
                                                                     convert_to_flat)


def test_manual_extraction():
    net = ResNet3D(in_channels=1)
    parameters, _ = extract_simple_resnet_parameters(net)
    parameters = convert_to_flat(parameters)

    for key, value in parameters.items():
        print(f'{key} :: {type(value)} :: {value.shape}')