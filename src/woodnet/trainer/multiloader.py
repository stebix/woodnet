"""
Implement a container for multiple DataLoaders that can be used in conjunction
with the multiloader trainer.
"""
import time
import logging
from collections.abc import Sequence

import numpy as np

from torch.utils.data import DataLoader

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger('main')


class MultiLoader:
    def __init__(
        self,
        loaders: Sequence[DataLoader],
        weights: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.loaders = loaders
        self.n_loaders = len(loaders)
        self.weights = self._init_weights(weights)
        self._seed = self._init_seed(seed)
        self._rng = np.random.default_rng(self._seed)
        self.reset()
        
    def _init_weights(self, weights: Sequence[float] | None) -> np.ndarray:
        if weights is None:
            return np.full(shape=self.n_loaders, fill_value=1/self.n_loaders)
        weights = np.array(weights)
        if not np.isclose(np.sum(weights), 1.0, atol=1e-4):
            raise ValueError(
                f'probability weights for the {self.n_loaders} must sum to '
                f'1, but got sum {np.sum(weights)}'
            )
        return weights

    def reset(self):
        """Reset all iterators and the active loader tracking."""
        self.iterators = [iter(loader) for loader in self.loaders]
        self.active_loaders = list(range(self.n_loaders))  # Indices of non-exhausted loaders
        self.active_weights = self.weights.copy()  # Copy of weights for active loaders
        
    def __next__(self):
        """
        Return the next batch from a randomly selected loader based on the weights.
        Raises StopIteration when all loaders are exhausted.
        """
        if not self.active_loaders:
            logger.debug('All loaders exhausted, raising StopIteration')
            raise StopIteration
            
        # Normalize active_weights to sum to 1
        if len(self.active_loaders) < self.n_loaders:
            active_sum = sum(self.active_weights[i] for i in self.active_loaders)
            if np.isclose(active_sum, 0.0):
                # If all remaining weights are effectively 0, use uniform distribution
                probs = np.ones(len(self.active_loaders)) / len(self.active_loaders)
            else:
                # Normalize remaining weights
                probs = np.array([self.active_weights[i] / active_sum for i in self.active_loaders])
        else:
            # Use original weights if all loaders are active
            probs = self.weights
            
        # Randomly select from active loaders
        idx = self._rng.choice(
            a=len(self.active_loaders),
            size=1,
            p=probs,
            replace=False
        ).item()
        loader_index = self.active_loaders[idx]
        
        logger.debug(f'loader index {loader_index} / {self.n_loaders} selected')
        
        try:
            batch = next(self.iterators[loader_index])
            logger.debug(f'loader index {loader_index} yielded batch')  #TODO: Remove in prod
            return batch
        except StopIteration:
            # Remove the exhausted loader from active_loaders
            logger.debug(f'loader index {loader_index} exhausted and removed from active loaders')
            self.active_loaders.remove(loader_index)
            
            # Try again if there are still active loaders
            if self.active_loaders:
                return next(self)
            else:
                logger.debug('All loaders exhausted, raising StopIteration')
                raise StopIteration

    def __iter__(self):
        """
        Return the iterator object (self).
        This method is required for the iterator protocol and enables
        the use of the MultiLoader in for loops and with enumerate().
        """
        # Reset iterators for a new iteration cycle
        self.reset()
        return self

    def __len__(self) -> int:
        """
        Return the number of batches in the multi-loader.
        This is the sum of the number of batches in each individual loader.
        """
        return sum(len(loader) for loader in self.loaders)


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


    def __repr__(self) -> str:
        repr_str = (
            f'{self.__class__.__name__}(n_loaders={self.n_loaders}, '
            f'lengths={[len(loader) for loader in self.loaders]}, '
            f'weights={self.weights})'
        )
        return repr_str
    
    def __str__(self) -> str:
        return self.__repr__()