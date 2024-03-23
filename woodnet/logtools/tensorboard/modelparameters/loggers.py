"""
Implement objects to programmatically log model parameters (i.e. weights and corresponding gradients)
to a tensorboard `SummaryWriter` backend 
"""
import sys
import logging
from torch.nn.modules import Module

from torch.utils.tensorboard.writer import SummaryWriter

from woodnet.logtools.tensorboard.modelparameters.extraction import extract_simple_resnet_parameters, convert_to_flat


DEFAULT_LOGGER_NAME: str = '.'.join(('main', __file__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


class VoidLogger:
    """
    Void logger that does not perform actual parameter extraction and reporting. 
    """
    def __init__(self) -> None:
        pass

    def log_weights(self, *args, **kwargs) -> None:
        pass

    def log_gradients(self, *args, **kwargs) -> None:
        pass

    def __str__(self) -> str:
        return repr(self)
        
    def __repr__(self) -> str:
        return ''.join((self.__class__.__name__, '()'))


class HistogramLogger:

    def __init__(self, writer: SummaryWriter) -> None:
        self.writer = writer

    def log_weights(self, model: Module, iteration: int) -> None:
        logger.debug(f'logging model weights at iteration {iteration}')
        nested_name_weights_mapping, _ = extract_simple_resnet_parameters(model)
        weights = convert_to_flat(nested_name_weights_mapping)
        for name, weightarray in weights.items():
            self.writer.add_histogram(tag=name, values=weightarray,
                                      global_step=iteration)

    def log_gradients(self, model: Module, iteration: int) -> None:
        logger.debug(f'logging model gradients at iteration {iteration}')
        logger.warning('currently dummy method')
        pass

    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(writer_log_dir=\'{self.writer.log_dir}\')'


def get_parameter_logger_class(name: str) -> type:
    """Retrieve parameter logger class by name."""
    modules = [sys.modules[__name__]]
    for module in modules:
        return getattr(module, name)
    raise AttributeError(f'could not retrieve parameter logger with name \'{name}\'')
