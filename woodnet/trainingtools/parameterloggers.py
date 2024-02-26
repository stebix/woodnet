"""
Implement objects to programmatically log model parameters (i.e. weights and corresponding gradients)
to a tensorboard `SummaryWriter` backend 
"""
import abc
import torch
from torch.nn.modules import Module

from torch.utils.tensorboard.writer import SummaryWriter 


class AbstractModelParameterLogger(abc.ABC):
    """
    Provides abstract architecture.
    """
    def __init__(self, writer: SummaryWriter) -> None:
        super().__init__()
        self.writer = writer

    @abc.abstractmethod
    def log_weights(self, model: torch.nn.Module, iteration: int) -> None:
        """Log the weights of the model to the `SummaryWriter`.
        """
        pass


    @abc.abstractmethod
    def log_gradients(self, model: torch.nn.Module, iteration: int) -> None:
        """Log the gradients of the model to the `SummaryWriter`
        """
        pass
    
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
        pass

    def log_gradients(self, model: Module, iteration: int) -> None:
        pass