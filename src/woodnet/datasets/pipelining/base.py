import logging

import numpy as np

from woodnet.datasets.pipelining.arrayseqmap import ArraySequence

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


class PipelineStep:
    log_action: bool = True
    """
    Base class for all pipeline steps.
    Pipeline steps are expected to be callable and to take a numpy array
    as input and return a numpy array as output.

    The `log_action` attribute determines whether the action of the step
    should be logged. 
    """
    pass


class BaseSubselector(PipelineStep):
    """
    Subselectors are subcalssed here as a semantic distinction from processors.
    They should ahere to the same interface as processors, but they are not
    expected to modify the data in any way other than selecting a subset of it.
    """
    log_action: bool = True

    def __call__(self, data: np.ndarray | ArraySequence) -> np.ndarray | ArraySequence:
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement __call__ method.'
        )

    def _emit_action_log(self, input: np.ndarray, output: np.ndarray) -> None:
        logger.debug(
            f'{str(self)} subselection action: {input.shape} -> {output.shape}'
        )


class BaseProcessor(PipelineStep):
    log_action: bool = True

    def __call__(self, data: np.ndarray | ArraySequence) -> np.ndarray | ArraySequence:
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement __call__ method.'
        )

    def _emit_action_log(self, input: np.ndarray, output: np.ndarray) -> None:
        logger.debug(
            f'{str(self)} processing action: {input.shape} -> {output.shape}'
        )