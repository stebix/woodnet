import numpy as np

from woodnet.models.planar import ResNet18
from woodnet.trainingtools.modelparameters.loggers import HistogramLogger


class Mockwriter:

    def __init__(self) -> None:
        pass

    def add_histogram(self, tag, values, global_step):
        # soft testing type of received argument values
        assert isinstance(tag, str)
        assert isinstance(values, np.ndarray)
        assert values.ndim == 1, f'expected flat array, but got ndim = {values.ndim}'
        assert isinstance(global_step, int)


def test_with_planar_ResNet18():
    model = ResNet18(in_channels=1)
    writer = Mockwriter()

    histlogger = HistogramLogger(writer=writer)

    for iteration in range(2):
        histlogger.log_weights(model=model, iteration=iteration)
        histlogger.log_gradients(model=model, iteration=iteration)

