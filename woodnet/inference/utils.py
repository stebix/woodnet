"""
general utility functions for inference purposes.

@jsteb 2024
"""
from typing import NamedTuple

class FilenameParts(NamedTuple):
    prefix: str
    qualifier: str | None
    UUID: str


def parse_checkpoint_fname(fname: str) -> FilenameParts:
    CHKPT_SUFFIX: str = '.pth'
    fname = fname.removesuffix(CHKPT_SUFFIX)
    try:
        prefix, qualifier, UUID = fname.split('_')
    except ValueError:
        prefix, UUID = fname.split('_')
        qualifier = None
    return FilenameParts(prefix, qualifier, UUID)
