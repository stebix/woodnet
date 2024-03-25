import uuid
import logging
import torch
import json

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

CHECKPOINT_FILE_SUFFIX: str = 'pth'
DEFAULT_PREFIX: str = 'chkpt'
SEP: str = '_'

def generate_filename(prefix: str | None = None, qualifier: str | None = None) -> str:
    """Generate filename for checkpoint"""
    prefix = prefix or DEFAULT_PREFIX
    if qualifier:
        mid = ''.join((SEP, qualifier, SEP))
    else:
        mid= SEP
    return f'{prefix}{mid}{uuid.uuid4()}.{CHECKPOINT_FILE_SUFFIX}'


def softcheck_filename(filepath: Path, /, action_verb: str = 'encountered') -> None:
    """
    Perform informal file naming scheme check: issue warning log messages if
    non-conforming.

    Parameters
    ==========

    filepath : Path
        Location on file system.

    action_verb: str, optional  
        Insert an approriate action verb to customize the log message.
        Defaults to the generic verb 'encountered'.
    """
    if not filepath.name.startswith(DEFAULT_PREFIX):
        logger.warning(f'{action_verb} checkpoint file with unusual name prefix: \'{filepath.name}\'')
    if not filepath.suffix.endswith(CHECKPOINT_FILE_SUFFIX):
        logger.warning(f'{action_verb} checkpoint file with unusual file suffix: \'{filepath.suffix}\'')


@dataclass
class RWDHandler:
    "Read - Write - Delete - Handler `CRDHandler` for checkpoint files."""
    directory: Path
    prefix: str | None = None
    serialization_target: Literal['state_dict', 'module'] = 'state_dict'

    def write(self,
              model: torch.nn.Module,
              qualifier: str | None = None,
              **kwargs) -> Path:
        """
        Write a model file (semantically named as a checkpoint) into the instance
        directory.

        Parameters
        ==========

        model: torch.nn.Module
            The model file.

        kwargs: Any
            Any further kwargs are forwarded tot the `torch.save` function.
        """
        filename = generate_filename(prefix=self.prefix, qualifier=qualifier)
        savepath = self.directory / filename
        softcheck_filename(savepath, action_verb='saving')
        if savepath.exists():
            raise FileExistsError(f'attempting to overwrite preexisting file at '
                                  f'\'{savepath.resolve()}\' with model file')
        torch.save(
            self.extract_serialization_target(model),
            f=savepath, **kwargs)
        logger.info(f'successfully saved checkpoint file to location \'{savepath}\'')
        return savepath


    def extract_serialization_target(self, model: torch.nn.Module):
        if self.serialization_target == 'state_dict':
            return model.state_dict()
        return model


    def write_json(self, data: Mapping, filename: str) -> Path:
        """Write data in Mapping-like format into a JSON file with the given filename."""
        savepath = self.directory / filename
        if savepath.exists():
            raise FileExistsError(f'cannot write JSON file to location \'{savepath}\' - '
                                  f'location already exists')
        with savepath.open(mode='w') as handle:
            json.dump(data, fp=handle)
        logger.info(f'successfully written JSON to location \'{savepath}\'')
        return savepath
        

    def check_working_directory(self, path: Path) -> None:
        if not path.parent == self.directory:
            raise ValueError(f'Requested operation outside of configured working '
                             f'directory. Working directory: \'{self.directory}\' and '
                             f'requested directory: \'{path.parent}\'')


    def read(self, name: str | None = None, path: Path | None = None, **kwargs) -> torch.nn.Module:
        """
        Load a checkpoint file by name or full path.
        Note that you can specify either the file name or the full path.

        Parameters
        ==========

        name: str, optional
            Load the pickled model file with the indicated file name.

        path: Path, optional
            Load the pickled model file from the full location path.

        **kwargs: Any
            Any further kwargs will be forwarded to the pytorch-internal
            load function.

        Returns
        =======

        model: torch.nn.Module
            The loaded model file.
        """
        if name and path:
            raise TypeError(f'parameters \'name\' and \'path\' with argument values '
                            f'({name}, {path}) are mutually exclusive')
        if name is None and path is None:
            raise TypeError('parameters \'name\' and \'path\' require at least one '
                            'not-None argument')
        if name:
            filepath = self.directory / name
        elif path:
            filepath = Path(path)

        self.check_working_directory(filepath)
        softcheck_filename(filepath, action_verb='reading')
        model = torch.load(f=filepath, **kwargs)
        logger.info(f'successfully loaded checkpoint file from location \'{filepath}\'')
        return model


    def delete(self, name: str | None = None, path: Path | None = None) -> None:
        """Delete the checkpoint file with the given path or name."""
        if name and path:
            raise TypeError(f'parameters \'name\' and \'path\' with argument values '
                            f'({name}, {path}) are mutually exclusive')
        if name is None and path is None:
            raise TypeError('parameters \'name\' and \'path\' require at least one '
                            'not-None argument')
        if name:
            filepath = self.directory / name
        elif path:
            filepath = Path(path)

        self.check_working_directory(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'could not delete non-existing file at location \'{filepath}\'')
        softcheck_filename(filepath, action_verb='deleting')
        filepath.unlink()
        logger.info(f'successfully deleted checkpoint file from location \'{filepath}\'')
    
