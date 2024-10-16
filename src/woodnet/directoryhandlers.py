import logging
import torch
import dataclasses

from pathlib import Path

from woodnet.custom.types import PathLike

LOGGER_NAME = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


@dataclasses.dataclass
class Directories:
    base: Path
    checkpoints: Path
    log: Path


def mkdir_logged(dirpath: Path, allow_preexisting: bool,
                 parents: bool = True, name: str = '') -> Path:
    """Log whether the indicated directory was created por already exsiting.

    Parameters
    ----------

    dirpath : Path
        Directory path.

    allow_preexisting : bool
        Allow preexsting directories, otherwise fail with `FileExistsError`
    
    parents : bool, optional
        Create parent directories if not preexisting.
        Defaults to False.

    name : str, optional
        Add indicative string of the direcory name or function to the log message.
        Defaults to empty string.


    Returns
    --------

    dirpath : Path
        Exiting directory path.

    Raises
    ------ 

    FileExistsError
        If the directory exists and the allow_preexisting flag is not set.
    """
    # pad name with one leading space if actually given to enable direct insert
    name = ' ' + name if name else name
    try:
        dirpath.mkdir()
    except FileExistsError as e:
        if allow_preexisting:
            logger.info(f'Using preexisting{name} directory \'{dirpath}\'')
            return dirpath
        else:
            raise e
    except FileNotFoundError as e:
        if parents:
            logger.info(f'Creating new{name} directory \'{dirpath}\' with deep parent tree')
            dirpath.mkdir(parents=True)
        else:
            raise e
    else:
        logger.info(f'Created new{name} directory \'{dirpath}\'')
    return dirpath
        



class ExperimentDirectoryHandler:
    """
    Handle the creation and management of experiment directories and provide facilities
    to save model checkpoints and logs to the respective directories.
    The class allows for fine-grained control over the directory creation and overwriting
    and emits log messages for each directory creation event.

    Parameters
    ----------

    directory : PathLike
        Base directory for the experiment. Experiment-related subdirectories will be
        created within this directory.

    Attributes
    ----------

    base : Path
        Base directory path for the experiment.

    logdir : Path
        Log directory path.

    checkpoints_dir : Path
        Checkpoints directory path.

    allow_preexisting_dir : bool
        Allow preexisting directories, otherwise fail with `FileExistsError`.
        For production use, this should be set to False to avert overwriting
        experiment results.

    allow_overwrite : bool
        Allow overwriting of files in the experiment directories.
        Again, for production use, this should be set to False to avert
        overwriting experiment results. For testing, this can be set to True
        to allow for easier testing of the directory handler.

    logdir_name : str
        Name of the log directory. Defaults to 'logs'.

    checkpointdir_name : str
        Name of the checkpoint directory. Defaults to 'checkpoints'.        
    """
    allow_preexisting_dir: bool = False
    allow_overwrite: bool = False

    logdir_name: str = 'logs'
    checkpointdir_name: str = 'checkpoints'

    def __init__(self,
                 directory: PathLike) -> None:
        
        dirs = self.initialize_directories(directory)
        self.base: Path = dirs.base
        self.logdir: Path = dirs.log
        self.checkpoints_dir: Path = dirs.checkpoints


    def initialize_directories(self, base: PathLike) -> None:
        """
        Initialize the experiment directory and its log and checkpoint sundirectories.
        Log messages are emitted for each directory creation event.
        """
        base = Path(base)
        mkdir_logged(base, self.allow_preexisting_dir, 'base experiment')
        log = base / self.logdir_name
        mkdir_logged(log, self.allow_preexisting_dir, 'log')
        checkpoints = base / self.checkpointdir_name
        mkdir_logged(checkpoints, self.allow_preexisting_dir, 'checkpoints')
        return Directories(base=base, checkpoints=checkpoints, log=log)

    
    def save_model_checkpoint(self, model: torch.nn.Module, name: str,
                              allow_overwrite: bool | None = None) -> None:
        """Save model checkpoint to the checkpoint directory.
        
        Parameters
        ----------

        model : torch.nn.Module
            Core model object.

        name : str
            Checkpoint file name, e.g. 'optimal.pth'

        allow_overwrite : bool or None, optional
            Set overwriting behaviour.
            Can be used to override instance-specific setting if
            expicit boolean is given. If set to None, then instance-wide
            setting is used.
            Defaults to None.
        """
        savepath = self.checkpoints_dir / name
        if savepath.exists() and not self.allow_overwrite:
            raise FileExistsError(f'preexisting checkpoint file at \'{savepath}\'')
        torch.save(model.state_dict(), savepath)
        logger.info(f'Saved model state dictionary to location \'{savepath}\'')