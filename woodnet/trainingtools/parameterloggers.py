"""
Implement objects to programmatically log model parameters (i.e. weights and corresponding gradients)
to a tensorboard `SummaryWriter` backend 
"""
import abc
import logging
import torch
from torch.nn.modules import Module

from torch.utils.tensorboard.writer import SummaryWriter

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __file__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


class StrReprMixin:
    """Provide str and repr dunder methods based on the `writer` attribute."""
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(writer_log_dir=\'{self.writer.log_dir}\')'


class AbstractWeightLogger(abc.ABC, StrReprMixin):
    """
    Report model weights to tensorboard.
    """
    def __init__(self, writer: SummaryWriter) -> None:
        super().__init__()
        self.writer = writer

    @abc.abstractmethod
    def log_weights(self, model: torch.nn.Module, iteration: int) -> None:
        pass


class AbstractGradientLogger(abc.ABC, StrReprMixin):
    """
    Report model weight gradients to tensorboard.
    """
    def __init__(self, writer: SummaryWriter) -> None:
        super().__init__()
        self.writer = writer

    @abc.abstractmethod
    def log_gradients(self, model: torch.nn.Module, iteration: int) -> None:
        pass


class AbstractModelParameterLogger(abc.ABC):
    """
    Provides abstract architecture.
    """
    def __init__(self, weightlogger: AbstractWeightLogger, gradientlogger: AbstractGradientLogger) -> None:
        super().__init__()
        self.gradientlogger = gradientlogger
        self.weightlogger = weightlogger
        self._check_sublogger_directories()


    def _check_sublogger_directories(self) -> None:
        weightlogger_dir = self.weightlogger.writer.log_dir
        gradlogger_dir = self.gradientlogger.writer.log_dir

        if not weightlogger_dir == gradlogger_dir:
            logger.warning(
                f'weight parameter sublogger (\'{weightlogger_dir}\') and gradient '
                f'parameter sublogger (\'{gradlogger_dir}\') log directories do not match'
            )


    @abc.abstractmethod
    def log_weights(self, model: torch.nn.Module, iteration: int) -> None:
        """Log the weights of the model to the `SummaryWriter`.
        """
        self.weightlogger.log_weights(model, iteration)


    @abc.abstractmethod
    def log_gradients(self, model: torch.nn.Module, iteration: int) -> None:
        """Log the gradients of the model to the `SummaryWriter`
        """
        self.gradientlogger.log_gradients(model, iteration)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(writer_log_dir=\'{self.writer.log_dir}\')'
    

class BasicModelParameterLogger(AbstractModelParameterLogger):
    """Log model parameters (wights and gradients) to the tensorboard `SummaryWriter`.

    This basic implementation separates the parameters layer-wise (basic ResNet schema)
    and leaves accumulation of values entirely to tensorboard (i.e. via histogram computation).
    """
    def log_weights(self, model: Module, iteration: int) -> None:
        logger.debug(f'logging model weights at iteration {iteration}')
        pass

    def log_gradients(self, model: Module, iteration: int) -> None:
        logger.debug(f'logging model gradients at iteration {iteration}')
        pass