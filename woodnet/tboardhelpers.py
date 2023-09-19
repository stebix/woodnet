"""
Small helpers to extract information from the tensorbaird event
file after the conclusion of the training phase.

Jannik Stebani 2023 
"""
import numpy as np
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

from woodnet.custom.types import PathLike

# hard coded shizzle to identify the event file when only
# the directory is given
LOGDIR_NAME: str = 'logs'
EVENTFILE_PREFIX: str = 'events.out.tfevents'

TRAINING_LOSS_TAG: str = 'loss/training_BCEWithLogitsLoss'
VALIDATION_LOSS_TAG: str = 'loss/validation_BCEWithLogitsLoss'


def retrieve_accumulator(path: PathLike) -> event_accumulator.EventAccumulator:
    """
    Load the event file and instantiate the accumulator from the indicated path.
    The path may point to the training directory or the event file itself.
    """ 
    path = Path(path)
    if path.is_file():
        return event_accumulator.EventAccumulator(path=str(path))
    elif path.is_dir():
        logdir = path / LOGDIR_NAME
        for item in logdir.iterdir():
            if item.name.startswith(EVENTFILE_PREFIX):
                return event_accumulator.EventAccumulator(path=str(item))
    raise FileNotFoundError(
        f'Could not find tensorboard event file at indicated location {path}'
    )


def retrieve_training_loss(path: PathLike) -> list[float]:
    accumulator = retrieve_accumulator(path)
    accumulator.Reload()
    return [event.value for event in accumulator.Scalars(TRAINING_LOSS_TAG)]


def retrieve_validation_loss(path: PathLike) -> list[float]:
    accumulator = retrieve_accumulator(path)
    accumulator.Reload()
    return [event.value for event in accumulator.Scalars(VALIDATION_LOSS_TAG)]


def retrieve_losses(path: PathLike) -> dict[str, list[float]]:
    return {
        'training' : retrieve_training_loss(path),
        'validation' : retrieve_validation_loss(path),

    }