"""
Facilities to programmatically interact with training directories for inference purposes.

@jsteb 2024
"""
import logging
import dataclasses

from collections.abc import Sequence, Iterable, Mapping
from pathlib import Path

from ruamel.yaml import YAML

from woodnet.utils import create_timestamp
from woodnet.directoryhandlers import mkdir_logged


CHECKPOINT_DIR_NAME: str = 'checkpoints'
LOG_DIR_NAME: str = 'logs'

TENSORBOARD_FILE_PREFIX: str = 'events.out.tfevents'
CHECKPOINT_FILE_SUFFIX: str = 'pth'
LOG_FILE_SUFFIX: str = 'log'
TRAINING_CONFIG_SUFFIX: str = 'yaml'

# logging infrastructure
LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)



def retrieve_optimal_checkpoint(checkpoints: Iterable[Path]) -> Path:
    for chkpt in checkpoints:
        if 'optimal' in chkpt.name:
            return chkpt
    raise LookupError('provided paths do not contain optimal checkpoint path')



def check_valid(p : Sequence[Path], /, type_: str, target: str = '') -> Path:
    """
    Check that the given sequence of paths is valid (unique and of correct type).

    Intended use: glob results sequence.

    Parameters
    ----------

    p : Sequence of Path
        Examinated sequence of paths. Usually the output of `glob.glob` or
        `pathlib.Path.glob`.

    type_ : {'file', 'dir'}
        Selector for the expected path type, i.e. a file or a directory.

    target : str, optional
        Provide additional information for the error message about the
        semantic meaning of the path. Defaults to empty string, i.e.
        no additional information.


    Returns
    -------

    p : Path
        Validated path.
    """
    p = tuple(p)
    if len(p) > 1:
        raise ValueError(f'Matched non-unique {type_} for target \'{target}\'. Matches: {p}')
    elif len(p) == 0:
        raise ValueError(f'No matching {type_} for target \'{target}\'.')
    p = p[0]
    if type_ == 'file':
        is_correct_type = p.is_file()
    elif type_ == 'dir':
        is_correct_type = p.is_dir()
    else:
        raise ValueError(f'Invalid type_ flag: \'{type_}\'. Must be \'file\' or \'dir\'')
    
    target = f' for target \'{target}\'' if target else ''
    if not is_correct_type:
        raise FileNotFoundError(f'matched object {p}{target} is not a {type_}')
    return p


def check_valid_file(p: Sequence[Path], target: str = '') -> Path:
    return check_valid(p, type_='file', target=target)


def check_valid_dir(p: Sequence[Path], target: str = '') -> Path:
    return check_valid(p, type_='dir', target=target)


def are_consecutive(nums: Sequence[int]) -> bool:
    nums = sorted(nums)
    start = min(nums)
    if nums == list(range(start, len(nums) + 1)):
        return True
    return False


def get_fold_directories(basepath: Path, sort: bool = True) -> dict[int, Path]:
    """Retrieve fold-like named subdirectories from the base directory."""
    directories = {}
    for item in basepath.iterdir():
        if item.is_dir() and item.name.startswith('fold-'):
            try:
                num = int(item.name.split('-')[1])
            except (ValueError, IndexError):
                continue
            directories[num] = item
    if sort:
        directories = {k : directories[k] for k in sorted(directories.keys())}
    return directories



@dataclasses.dataclass
class TrainingResultBag:
    """
    Interface to programmatically interact with validated training results directories.
    """
    # directory paths
    basepath: Path
    checkpoint_directory: Path
    log_directory: Path
    # relevant file paths
    checkpoints: list[Path]
    optimal_checkpoint: Path
    logfile: Path
    training_configuration: Path
    tensorboard_file: Path

    def load_configuration(self) -> dict:
        """
        Load the configuration YAML file from the log directory.
        """
        yaml = YAML()
        with self.training_configuration.open(mode='r') as handle:
            configuration = yaml.load(handle)
        return configuration
            
    
    @classmethod
    def from_directory(cls, basepath: Path) -> 'TrainingResultBag':
        """
        Create instance from training top-level directory.
        Performs soft validations for the existence of all directories and files.

        Directories: {checkpoint directory, log directory}
        Files: {Checkpoint files, optimal checkpoint file,
                training configuration backup file, log file,
                tensorboard log file}
        """
        checkpoint_dir = check_valid_dir(tuple(basepath.glob(CHECKPOINT_DIR_NAME)),
                                         'checkpoint dir')
        log_dir = check_valid_dir(tuple(basepath.glob(LOG_DIR_NAME)), 'log dir')
        checkpoints  = list(checkpoint_dir.glob('.'.join(('*', CHECKPOINT_FILE_SUFFIX))))
        optimal_checkpoint = retrieve_optimal_checkpoint(checkpoints)
        logfile = check_valid_file(tuple(log_dir.glob('.'.join(('*', LOG_FILE_SUFFIX)))), 'log file')
        training_conf = check_valid_file(
            tuple(log_dir.glob('.'.join(('*', TRAINING_CONFIG_SUFFIX)))),
            'training configuration'
        )
        tensorboard_file = check_valid_file(
            tuple(log_dir.glob('.'.join((TENSORBOARD_FILE_PREFIX, '*')))),
            'tensorboard file'
        )
        return cls(basepath=basepath,checkpoint_directory=checkpoint_dir,
                   log_directory=log_dir, checkpoints=checkpoints,
                   optimal_checkpoint=optimal_checkpoint, logfile=logfile,
                   training_configuration=training_conf,
                   tensorboard_file=tensorboard_file)



@dataclasses.dataclass
class CrossValidationResultsBag:
    basepath: Path
    folds: Mapping[int, TrainingResultBag]
    foldcount: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.foldcount = len(self.folds)
        logger.debug(
            f'Successfully parsed \'{self.basepath}\' as {self.__class__.__name__} '
            f'with {self.foldcount} folds.'
        )

    @classmethod
    def from_directory(cls, basepath: Path) -> 'CrossValidationResultsBag':
        """
        Auto initialize from appropriately structured directories.
        Fold-wise subdirectories are cast and validated as `TrainingResultsBag`.
        Expected layout:

            basedir
                |--- fold-1
                |      |---- checkpoints
                |      |---- logs
                |
                |--- fold-2
                |      |---- checkpoints
                |      |---- logs
                ...
                |--- fold-N
                       |---- checkpoints
                       |---- logs
        """
        folds_mapping = get_fold_directories(basepath)
        folds = {}
        for foldnum, fold_dir in folds_mapping.items():
            # every fold-wise subdirectory is expected to be a valid
            # training results directory
            folds[foldnum] = TrainingResultBag.from_directory(fold_dir)
        
        if not are_consecutive(folds.keys()):
            logger.warning(f'Cross validation folds in base directory \'{basepath}\''
                           f'are non-consecutive. Fold nums: {folds.keys()}')
        
        return cls(basepath=basepath, folds=folds)
    

    def retrieve_model_instances_locations(self) -> dict[int, list[Path]]:
        """
        Retrieve the paths to all model instances (checkpoints) for all folds.
        
        Returns
        -------

        dict[int, list[Path]]
            Dict mapping from the cross validation fold number to the list of paths
            pointing to the model instances, i.e. optimal model checkpoints.
        """
        return {fnum : results.checkpoints for fnum, results in self.folds.items()}



def create_inference_directory(basedir: Path) -> Path:
    """
    Create an inference run directory.
    """
    timestamp = create_timestamp()
    inference_dir = basedir / timestamp
    inference_dir = mkdir_logged(inference_dir, allow_preexisting=False, parents=True,
                                 name='concrete inference')
    return inference_dir


