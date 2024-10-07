"""
General configuration handling tooling.

@jsteb 2023
"""
import shutil
import warnings
import pathlib
from collections.abc import Mapping

from ruamel.yaml import YAML

from woodnet.custom.types import PathLike
from woodnet.configtools.validation import TrainingConfiguration
from woodnet.directoryhandlers import ExperimentDirectoryHandler
from woodnet.utils import backup_file

# settings
CUDA_MAX_DEVICE_INDEX: int = 3

ConfigurationLike = Mapping | TrainingConfiguration


def load_yaml(path: PathLike) -> Mapping:
    """Load content of th YAML file from the indicated location."""
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    yaml = YAML()
    with path.open(mode='r') as handle:
        content = yaml.load(handle)

    return content


def write_yaml(content: Mapping, path: PathLike,
               force_write: bool = False) -> pathlib.Path:
    """Write mappiong-type data as YAMl to the file system."""
    path = pathlib.Path(path)
    if path.exists() and not force_write:
        raise FileExistsError(f'Cannot write to location \'{path}\'. Select new path or '
                              f'force write via flag argument.')
    yaml = YAML()
    with path.open(mode='w') as handle:
        yaml.dump(content, stream=handle)
    return path


def backup_configuration(source: PathLike | dict,
                         target: PathLike | ExperimentDirectoryHandler,
                         force_write: bool = False) -> pathlib.Path:
    """Backup a configuration file or object to the indicated target location."""
    if isinstance(source, PathLike):
        target = backup_file(source, target, force_write)
    else:

        if isinstance(target, PathLike):
            target = pathlib.Path(target)
            if not target.suffix not in {'.yaml', '.yml'}:
                warnings.warn(f'Selected target path for the configuration \'{target}\' '
                              f'has no YAML file suffix!')
        else:
            target = target.logdir / 'backup_training_configuration.yaml'
        write_yaml(source, target, force_write)
    return target


def update_cuda_device_index(configuration: ConfigurationLike, device_index: int) -> None:
    """
    In-place update the CUDA device index.

    Parameters
    ----------

    congiuration : ConfigurationLike
        Full configuration with device attribute at the top level.

    device_index : int
        Target device index.
    """
    if device_index > CUDA_MAX_DEVICE_INDEX:
        warnings.warn(f'given CUDA device index {device_index} exceeds '
                      f'CUDA_MAX_DEVICE_INDEX = {CUDA_MAX_DEVICE_INDEX}')

    if isinstance(configuration, Mapping):
        if 'cuda' not in configuration['device']:
            warnings.warn(f'current device appear to be non-CUDA: \'{configuration["device"]}\'')
        configuration['device'] = f'cuda:{device_index}'
    
    elif isinstance(configuration, TrainingConfiguration):
        if 'cuda' not in configuration.device:
            warnings.warn(f'current device appear to be non-CUDA: \'{configuration["device"]}\'')
        configuration.device = f'cuda:{device_index}'
