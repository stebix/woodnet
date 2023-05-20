from pathlib import Path
import json
import numpy as np

from collections import defaultdict
from typing import Optional

from loader import parse_filename_identifier


DEFAULT_SUFFIX: str = 'tif'
DEFAULT_SUBVOLUME_PREFIX: str = 'subvol'


def build_output_filename(ID: int, class_: str,
                          subvolume: Optional[int] = None) -> str:
    """Build output JSON file name from data attributes."""
    if subvolume:
        subvolume_info = f'subvolume-{subvolume}'
    else:
        subvolume_info = 'complete'
    return f'CT-{ID}_{class_}_{subvolume_info}_statistics.json'


def is_subvolumedirectory(p: Path) -> bool:
    pattern = ''.join(('/', DEFAULT_SUBVOLUME_PREFIX, '_*/*'))
    return p.match(pattern)


def contains_many_tiffs(p: Path, at_least: int = 25) -> bool:
    """Soft validation insinuating that we are in a dataset directory."""
    tiff_count = 0
    for child in p.iterdir():
        if child.name.endswith(DEFAULT_SUFFIX):
            tiff_count += 1
        if tiff_count >= at_least:
            return True
    return False


def collect_data_directories(basedir: Path) -> dict[str, list]:
    directories = defaultdict(list)
    # recursive walk into all directories
    for child in basedir.glob('**/**/'):
        if is_subvolumedirectory(child) and contains_many_tiffs(child):
            directories['subvolume'].append(child)
        elif contains_many_tiffs(child):
            directories['complete'].append(child)
        else:
            directories['indeterminate'].append(child)
    return directories



if __name__ == '__main__':
    pass