import os
import logging

from dataclasses import dataclass
from numbers import Number
from pathlib import Path

from woodnet.checkpoint.scores import ScoreRank
from woodnet.checkpoint.handlers import SEP

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


def excise_optimal_qualifier(path: Path) -> Path:
    filename = path.name
    prefix, qualifier, uuid_and_suffix = filename.split(SEP)
    if qualifier != 'optimal':
        logger.warning(f'excised unusual qualifier \'{qualifier}\' '
                       f'from path \'{path}\'')
    return path.parent / SEP.join((prefix, uuid_and_suffix))


@dataclass
class ScoredCheckpoint:
    filepath: Path
    score: Number
    rank: ScoreRank

    def remove(self) -> None:
        """Remove the model checkpoint file from the file system."""
        try:
            self.filepath.unlink(missing_ok=False)
        except FileNotFoundError:
            logger.warning(f'attempted to remove non-existing checkpoint file at '
                           f'\'{self.filepath}\'')
            return
        logger.info(f'removed checkpoint file at \'{self.filepath}\'')


    def demote(self) -> None:
        """
        Demote the checkpoint to non-optimal feasible status.
        Renames checkpoint file.
        Issues warning log messages if non-optimal ScoredCheckpoint instance
        is attempted to be demoted. 
        """
        if self.rank is not ScoreRank.OPTIMAL:
            message = (f'attempted demotion of non-optimal instance '
                       f'with score rank of {self.rank}')
            logger.warning(message)
        
        try:
            demoted_filepath = excise_optimal_qualifier(self.filepath)
        except ValueError:
            logger.warning(f'Could not demote scored checkpoint at \'{self.filepath}\'. '
                           f'Qualifier excision from filename failed.')
            return
        
        os.rename(self.filepath, demoted_filepath)
        logger.info(f'Successfully demoted formerly optimal checkpoint @ \'{self.filepath}\' '
                    f'to feasible checkpoint @ \'{demoted_filepath}\'')
        self.filepath = demoted_filepath
        self.rank = ScoreRank.FEASIBLE
        return