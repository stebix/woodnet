"""
Facilities for torch DataLoaders with specific parametrized transform
and ablation experiment applications in mind.

@jsteb 2024
"""
import os
import logging
import tqdm

from collections.abc import Callable
from functools import wraps
from torch.utils.data import DataLoader, Dataset


LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def deduce_display_progressbar() -> bool:
    env_value = os.environ.get('DISPLAY_PROGRESSBAR', None)
    logger.debug(f'retrieved environment value: DISPLAY_PROGRESSBAR = \'{env_value}\'')
    falsy: set[str] = {'0', 'false', 'False'}
    return env_value not in falsy

# make progress reporting steerable via environment variables
DISPLAY_PROGRESSBAR: bool = deduce_display_progressbar()
LEAVE_PROGRESSBAR: bool = False

def generate_loader_function_factory(display_progress: bool, leave_progress_display: bool) -> Callable:
    """
    Generates the `generate_loader` function depending on the deduced progress reporting settings.
    """
    if not display_progress:
        def _generate_loader(dataset: Dataset, batch_size: int) -> DataLoader:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return _generate_loader
    settings = {'unit' : 'bt', 'leave' : leave_progress_display, 'desc' : 'dataloader'}
    def _generate_loader(dataset: Dataset, batch_size: int) -> DataLoader:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        wrapped = tqdm.tqdm(loader, **settings)
        return wrapped
    return _generate_loader

BatchSize = int

generate_loader: Callable[[Dataset, BatchSize], DataLoader] = generate_loader_function_factory(DISPLAY_PROGRESSBAR, LEAVE_PROGRESSBAR)