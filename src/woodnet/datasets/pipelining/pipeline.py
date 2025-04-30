import numpy as np

from woodnet.datasets.pipelining.base import PipelineStep

class Pipeline:

    def __init__(
        self,
        *steps: PipelineStep,
    ) -> None:
        self.steps = steps

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the pipeline to the data.
        """
        for step in self.steps:
            data = step(data)
        return data