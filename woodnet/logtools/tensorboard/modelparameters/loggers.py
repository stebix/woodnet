"""
Implement objects to programmatically log model parameters (i.e. weights and corresponding gradients)
to a tensorboard `SummaryWriter` backend 
"""
import sys
import logging
from collections.abc import Mapping

from torch.nn.modules import Module
from torch.utils.tensorboard.writer import SummaryWriter

from woodnet.logtools.tensorboard.modelparameters.extraction import (extract_simple_resnet_parameters,
                                                                     extract_simple_resnet_gradients,
                                                                     convert_to_flat)
from woodnet.logtools.dict import LoggedDict


DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

DEFAULT_PARAMETER_LOGGER_NAME: str = 'VoidLogger'

class VoidLogger:
    """
    Void logger that does not perform actual parameter extraction and reporting. 
    """
    def __init__(self, *args, **kwargs) -> None:
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
        self._is_first_weightlog = True
        self._is_first_gradlog = True


    def log_weights(self, model: Module, iteration: int) -> None:
        """Log gradients of ResNet model to tensorboard."""
        if self._is_first_weightlog:
            logger.debug(f'first logging weights with model {model.__class__.__name__}')
            self._is_first_weightlog = False

        if hasattr(model, '_orig_mod'):
            logger.debug('Detected torch JIT-compiled model, using \'_orig_mod\' atribute.')
            model = model._orig_mod

        logger.debug(f'logging model weights at iteration {iteration}')
        nested_name_weights_mapping, _ = extract_simple_resnet_parameters(model)
        weights = convert_to_flat(nested_name_weights_mapping)

        logger.debug(f'retrieved weights keys {weights.keys()}')
        logger.debug(f'retrieved unrecognized {_}')


        for name, weightarray in weights.items():
            self.writer.add_histogram(tag=name, values=weightarray,
                                      global_step=iteration)

            # TODO: REMOVE
            logger.debug(f'add_histogram call with tag \'{name}\', '
                         f'values shape {weightarray.shape} and iteration {iteration}')



    def log_gradients(self, model: Module, iteration: int) -> None:
        """Log gradients of ResNet model to tensorboard."""
        if self._is_first_gradlog:
            logger.debug(f'first logging gradients with model {model.__class__.__name__}')
            self._is_first_gradlog = False

        if hasattr(model, '_orig_mod'):
            logger.debug('Detected torch JIT-compiled model, using \'_orig_mod\' atribute.')
            model = model._orig_mod

        logger.debug(f'logging model gradients at iteration {iteration}')
        nested_name_gradients_mapping, _ = extract_simple_resnet_gradients(model)
        weights = convert_to_flat(nested_name_gradients_mapping)

        logger.debug(f'retrieved weights keys {weights.keys()}')
        logger.debug(f'retrieved unrecognized {_}')

        for name, weightarray in weights.items():
            self.writer.add_histogram(tag=name, values=weightarray,
                                      global_step=iteration)
            # TODO: REMOVE
            logger.debug(f'add_histogram call with tag \'{name}\', '
                         f'values shape {weightarray.shape} and iteration {iteration}')


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


def create_parameter_logger(configuration: Mapping | None,
                            writer: SummaryWriter) -> VoidLogger | HistogramLogger:
    """
    Create a parameter logger from the subconfiguration.
    """
    configuration = LoggedDict(configuration or {}, logger)
    name = configuration.get('name', DEFAULT_PARAMETER_LOGGER_NAME)
    class_ = get_parameter_logger_class(name)
    return class_(writer)
