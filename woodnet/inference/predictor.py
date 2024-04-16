"""
Programmatic interaction with prediction tasks in the context of robustness
experiments.

@jsteb 2024
"""
import contextlib
import logging
from collections.abc import Sequence

import torch
import torch.utils
import tqdm

from woodnet.inference.evaluate import evaluate
from woodnet.inference.inference import ParametrizationsContainer


LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)

