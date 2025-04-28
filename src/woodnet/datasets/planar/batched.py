import logging
import time

from collections.abc import Callable
from typing import Literal
from functools import cached_property

import numpy as np
import torch

from woodnet.datasets.fingerprints import Fingerprint, StatParams
from woodnet.datasets.planar.slicebased import TileDataset

from woodnet.datasets.batching import generate_batch_indices, filter_pad_value

Tensor = torch.Tensor

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


class BatchedTileDataset(TileDataset):
    """
    2D TileDataset that is statically batched,
    i.e. batching is done in the __getitem__ method.
    """
    length_lower_warning_limit: int = 2
    def __init__(
        self,
        phase: Literal['train', 'val', 'test'],
        data: np.ndarray,
        fingerprint: Fingerprint,
        stats: StatParams,
        transformer: Callable[[torch.Tensor], torch.Tensor] = None,
        classlabel_mapping: dict[str, int] = None,
        batch_size: int = 1,
        drop_last: bool = False,
        filter_batch_padding: bool = True,
        seed: int | None = None,
    ) -> None:
        
        self.phase: Literal['train', 'val', 'test'] = phase
        self.shape = None
        self.channels: int = 0
        self.data: np.ndarray = self._initialize_data(data)
        self.transformer = transformer
        self.fingerprint: Fingerprint = fingerprint
        self.stats: StatParams = stats
        self.classlabel_mapping= classlabel_mapping or {}
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.filter_batch_padding = filter_batch_padding
        self._seed = self._init_seed(seed)
        self._rng = np.random.default_rng(self._seed)
        self.batch_indices = self._init_indices(batch_size)
        self._shuffle_counter: int = 0


    def _initialize_data(self, data: np.ndarray) -> np.ndarray:
        """Determine the data shape and add a fake channel dimension if needed."""
        if data.ndim == 3:
            self.shape = data.shape
            logger.debug(f'Adding fake channel dimension to data shape: {data.shape}')
            # add a fake channel dimension
            self.channels = 1
            return np.expand_dims(data, axis=0)
        elif data.ndim == 4:
            # data already has a channel dimension
            C, *shape = data.shape
            self.shape = shape
            self.channels = C
            return data

        raise ValueError(
            f'Invalid data shape: {data.shape}. Expected 3D or 4D array '
            f'with layout (C, D, H, W) or (D, H, W) respectively.'
        )

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
    

    def _init_indices(self, batch_size: int) -> np.ndarray:
        elements = self.shape[0]
        logger.debug(
            f'Initializing indices for {elements} elements and batch size {batch_size} '
            f'with basal data shape {self.shape}'
        )
        batch_indices = generate_batch_indices(
            n_elements=elements,
            batch_size=batch_size,
            seed=self._seed,
            drop_last=self.drop_last
        )
        return batch_indices
        

    def shuffle_batch_indices(self) -> None:
        """
        In-place shuffling of the batch indices.
        Inteded at epoch end to mix up intra-dataset batch order.
        """
        logger.debug(
            f'{self.__class__.__name__} shuffling indices'
        )
        self._rng.shuffle(self.batch_indices)
        self._shuffle_counter += 1


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | Tensor:
        indices = self.batch_indices[index]

        if self.filter_batch_padding and not self.drop_last:
            # filter out padding values
            indices = filter_pad_value(indices)

        # elements
        batch_elements = [torch.tensor(self.data[:, i, :, :]) for i in indices]
        if self.transformer is not None:
            batch_elements = [self.transformer(e) for e in batch_elements]

        batch_data_tensor = torch.stack(batch_elements, dim=0)

        if self.phase == 'test':
            # apply normalization
            return batch_data_tensor

        effective_batch_size = len(batch_elements)
        batch_label_tensor = torch.broadcast_to(
            torch.tensor(self.label).unsqueeze_(0).unsqueeze_(0),
            size=(effective_batch_size, 1)
        )
        return (batch_data_tensor, batch_label_tensor)


    def __len__(self) -> int:
        assert self.batch_indices.ndim == 2, (f'batch indices malformed with ndim'
                                              f'={self.batch_indices.ndim}D')
        return self.batch_indices.shape[0]
    
    @property
    def planeshape(self) -> tuple[int, int]:
        return self.shape[1:]
    
    @property
    def volumeshape(self) -> tuple[int, int, int]:
        return self.shape

    @cached_property
    def class_(self) -> str:
        return self.fingerprint.class_

    @cached_property
    def label(self) -> int:
        return self.classlabel_mapping[self.fingerprint.class_]
    
    @property
    def shuffle_counter(self) -> int:
        return self._shuffle_counter

    def _check_dataset_length_(self) -> None:
        if len(self) < self.length_lower_warning_limit:
            # log a warning if the dataset length is lower than the limit
            # but do not raise an error
            logger.warning(
                f'{self.__class__.__name__} length={len(self)} is lower than '
                f'length_lower_warning_limit={self.length_lower_warning_limit}'
            )    

    def _log_initialization(self) -> None:
        logger.info(
            f'Initialization success for {self.__class__.__name__} with phase=\'{self.phase}\' '
            f'shape={self.shape} channels={self.channels} and '
            f'length={len(self)} and class_=\'{self.class_}\' and label={self.label} '
            f'and batch size={self.batch_size} and drop_last={self.drop_last} '
            f'and filter_batch_padding={self.filter_batch_padding} '
        )
        self._check_dataset_length_()
