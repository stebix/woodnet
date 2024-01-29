import logging

from dataclasses import dataclass
from numbers import Number
from pathlib import Path

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


@dataclass(frozen=True)
class ScoredCheckpoint:
    filepath: Path
    score: Number

    def remove(self) -> None:
        """Remove the model checkpoint file from the file system."""
        try:
            self.filepath.unlink(missing_ok=False)
        except FileNotFoundError:
            logger.warning(f'attempted to remove non-existing checkpoint file at '
                           f'\'{self.filepath}\'')
            pass
        logger.info(f'removed checkpoint file at \'{self.filepath}\'')