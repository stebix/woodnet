"""
Facilities to configure global PyTorch settings that require to be set
before any actual eager or JIT operations.
Examples: torch cpu interop threads or torch-internal logging

@jsteb 2024
"""
import os
import logging
import torch

from collections.abc import Callable
from typing import Any

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


_NOTSET = object()


def deduce_from_env(name: str, default: Any, cast: Callable) -> Any:
    """Deduce and cast values via string names from evironment.
    
    Emits log messages.
    """
    raw = os.environ.get(name, default=_NOTSET)
    if raw is _NOTSET:
        logger.debug(f'Environment does not define \'{name}\'. Utiliying '
                     f'hardcoded fallback \'{default}\'')
        value = default
    else:
        value = cast(raw)
        logger.debug(f'Retrieved raw value {name} = \'{raw}\' from '
                     f'environment. Utilizing cast result: \'{value}\'')
    return value


DEFAULT_THREAD_COUNT: int = 16
TORCH_NUM_THREADS: int = deduce_from_env('TORCH_NUM_THREADS',
                                         default=DEFAULT_THREAD_COUNT,
                                         cast=int)
TORCH_NUM_INTEROP_THREADS: int = deduce_from_env('TORCH_NUM_INTEROP_THREADS',
                                                 default=DEFAULT_THREAD_COUNT,
                                                 cast=int)


def configure_torch_cpu_threading(num_threads: int | None,
                                  num_interop_threads: int | None) -> None:
    """
    Log current and set desired PyTorch CPU threading setting.
    Acts as simple logger without anys etting action if both thread counts are `None`.
    """
    logger.debug(f'Current setting TORCH_NUM_THREADS = {torch.get_num_threads()}')
    logger.debug(f'Current setting TORCH_NUM_INTEROP_THREADS = {torch.get_num_interop_threads()}')
    if num_threads:
        logger.debug(f'Changing to user value TORCH_NUM_THREADS = {num_threads}')
        torch.set_num_threads(num_threads)
    if num_interop_threads:
        logger.debug(f'Changing to user value TORCH_NUM_INTEROP_THREADS = {num_interop_threads}')
        torch.set_num_threads(num_interop_threads)
    return None
