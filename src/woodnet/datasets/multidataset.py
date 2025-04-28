import logging
import time

from collections.abc import Sequence

import torch

from torch.utils.data.dataset import ConcatDataset, Dataset

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger('main')


class GroupDataset(ConcatDataset):
    """
    A dataset that encapsulates a group of datasets.
    Due to the group cahracteristics, this kind of dataset
    can generally not be mixed with other datasets.
    Thus, batching is handled by the dataset itself.
    """
    def __init__(
        self,
        datasets: Sequence[Dataset],
        batch_size: int,
        seed: int | None = None,
    ) -> None:
        super().__init__(datasets)
        self.batch_size = batch_size
        self._seed = self._init_seed(seed)


    def _init_seed(self, seed: int | None) -> int:
        """
        Initialize the seed for random number generation.
        If no seed is provided, a random seed is generated.
        """
        if seed is None:
            seed = int(time.time())
            logger.debug(f'{self.__class__.__name__} using time based RNG seed {seed}')
        else:
            logger.debug(f'{self.__class__.__name__} using provided RNG seed {seed}')
        return seed


    def __getitem__(self, index: int) -> tuple:
        pass