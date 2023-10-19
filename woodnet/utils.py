import datetime
import pathlib
import shutil

from collections.abc import Iterable, Mapping

from woodnet.custom.types import PathLike
from woodnet.directoryhandlers import ExperimentDirectoryHandler

DEFAULT_TIMESTAMP_FORMAT: str = '%Y-%m-%d_%H-%M-%S'

def create_timestamp(fmt: str | None = None) -> str:
    """Curent timestamp with second-wise accuracy."""
    fmt = fmt or DEFAULT_TIMESTAMP_FORMAT
    return datetime.datetime.now().strftime(fmt)


def generate_keyset(dicts: Iterable[Mapping]) -> set:
    """Generate the set of key values."""
    s = set()
    for d in dicts:
        s = s | d.keys()
    return s


def backup_file(source: PathLike,
                target: PathLike | ExperimentDirectoryHandler,
                force_write: bool = False) -> pathlib.Path:
    """Backup a file by copying it from the source to the target location."""
    if isinstance(target, PathLike):
        target = pathlib.Path(target)
    else:
        target = target.logdir

    if target.is_file() and not force_write:
        raise FileExistsError(f'Cannot backup file to \'{target}\ - already existing.')
    
    return shutil.copy(source, target)
