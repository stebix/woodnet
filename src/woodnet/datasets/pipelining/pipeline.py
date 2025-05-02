import numpy as np

from woodnet.datasets.pipelining.base import PipelineStep
from woodnet.datasets.pipelining.arrayseqmap import ArraySequence

class Pipeline:

    def __init__(
        self,
        *steps: PipelineStep,
    ) -> None:
        self.steps = steps

    def __call__(self, data: np.ndarray | ArraySequence) -> np.ndarray | ArraySequence:
        """
        Apply the pipeline to the data.
        """
        for step in self.steps:
            data = step(data)
        return data
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        format_string += '(\n'
        format_string += f'  step_count={len(self.steps)},\n'
        format_string +=  '  steps=\n    ['
        for step in self.steps:
            format_string += '\n'
            format_string += f'        {step}'
        format_string += '\n    ]\n)'
        return format_string

    def __str__(self) -> str:
        return repr(self)