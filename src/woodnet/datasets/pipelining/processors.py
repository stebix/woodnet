import logging

from typing import Literal

import numpy as np

from woodnet.datasets.pipelining.base import BaseProcessor


DEFAULT_LOGGER_NAME: str = '.'.join('main', __name__)
logger: logging.Logger = logging.getLogger(DEFAULT_LOGGER_NAME)


class ChannelSqueezingProcessor(BaseProcessor):
    """
    Subselector that squeezes the channel dimension of the data.
    For multichannel data, we can specify a strategy to handle the extra channels
    by raising an error, warning, or selecting a specific channel.
    The ouput data will have the shape (D, H, W) if the input data has the shape (C, D, H, W).
    """
    def __self__(
        self,
        multichannel_strategy: Literal['raise', 'warn', 'select'] = 'raise',
        channel_selection: int | None = None, 
    ) -> None:
        self.multichannel_strategy = multichannel_strategy
        self.channel_selection = channel_selection

    def __call__(self, data: np.ndarray) -> np.ndarray:
        *pre, C, D, H, W = data.shape
        pre = tuple(slice(None) for _ in range(len(pre)))
        if C == 1:
            processed = data[*pre, 0, :, :, :]
            if self.log_action:
                self._emit_action_log(data, processed, cidx=0)
            return processed
        if C > 1:
            if self.multichannel_strategy == 'raise':
                raise ValueError(
                    f'Input data has {C} channels, but only one channel is expected. '
                    f'Use multichannel_strategy=\'select\' to select a channel.'
                )
            elif self.multichannel_strategy == 'warn':
                logger.warning(
                    f'Input data has {C} channels, but only one channel is expected. '
                    f'Using channel index 0'
                )
                c_idx = 0
            elif self.multichannel_strategy == 'select':
                if self.channel_selection is None:
                    raise ValueError(
                        'channel_selection must be provided when multichannel_strategy=\'select\''
                    )
                c_idx = self.channel_selection
            
            processed = data[*pre, c_idx, :, :, :]
            if self.log_action:
                self._emit_action_log(data, processed, cidx=c_idx)
            return processed


    def _emit_action_log(self, input: np.ndarray, output: np.ndarray, cidx: int) -> None:
        logger.debug(
            f'{str(self)} channel processing action: {input.shape} -> {output.shape} '
            f'with channel index {cidx}'
        )