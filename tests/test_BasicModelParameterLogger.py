import numpy as np
import torch

from woodnet.models.planar import ResNet18
from woodnet.logtools.tensorboard.modelparameters.loggers import HistogramLogger


class Mockwriter:

    def __init__(self) -> None:
        pass

    def add_histogram(self, tag, values, global_step):
        # soft testing type of received argument values
        assert isinstance(tag, str)
        assert isinstance(values, np.ndarray)
        assert values.ndim == 1, f'expected flat array, but got ndim = {values.ndim}'
        assert isinstance(global_step, int)


def test_weight_logging_with_planar_ResNet18():
    model = ResNet18(in_channels=1)
    writer = Mockwriter()

    histlogger = HistogramLogger(writer=writer)

    for iteration in range(2):
        histlogger.log_weights(model=model, iteration=iteration)



def test_gradient_logging_with_planar_ResNet18():
    model = ResNet18(in_channels=1)
    model.to(torch.float32)
    writer = Mockwriter()

    histlogger = HistogramLogger(writer=writer)

    # Here we have to populate the gradients manually.
    # In its default state, the .grad attribute is set to `None`.
    SHAPE = (1, 1, 128, 128)
    input = torch.randn(SHAPE)
    noisy_input = input + torch.randn_like(input)
    output = model(input)
    error = torch.sum(torch.abs(output - noisy_input))
    error.backward()

    for iteration in range(2):
        histlogger.log_gradients(model=model, iteration=iteration)
