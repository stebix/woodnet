import json
import dataclasses
import numpy as np

from pathlib import Path
from collections import defaultdict
from typing import Optional, Protocol

from dataobjects import InstanceFingerprint, SubvolumeFingerprint
from custom.types import PathLike


DEFAULT_SUFFIX: str = 'tif'
DEFAULT_SUBVOLUME_PREFIX: str = 'subvol'


class VolumeLike(Protocol):
    fingerprint: InstanceFingerprint
    data: np.ndarray


def build_output_filename(ID: int, class_: str,
                          subvolume: Optional[int] = None) -> str:
    """Build output JSON file name from data attributes."""
    if subvolume:
        subvolume_info = f'subvolume-{subvolume}'
    else:
        subvolume_info = 'complete'
    return f'CT-{ID}_{class_}_{subvolume_info}_statistics.json'


def build_output_filename_from(fingerprint: InstanceFingerprint) -> str:
    if isinstance(fingerprint, SubvolumeFingerprint):
        subvolume = fingerprint.index
    else:
        subvolume = None
    filename = build_output_filename(
        ID=fingerprint.ID, class_=fingerprint.class_, subvolume=subvolume
    )
    return filename


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


def compute_statistics(volume: VolumeLike) -> dict:
    statistics = {
        'metadata' : {
            'fingerprint' : str(volume.fingerprint.__class__.__name__),
            **dataclasses.asdict(volume.fingerprint)
        }
    }
    # Cast to float to make the values JSON-serializable.
    # 32 bit numpy floats are not natively JSON-serializable
    statistics['minimum'] = float(np.min(volume.data))
    statistics['maximum'] = float(np.max(volume.data))
    statistics['mean'] = float(np.mean(volume.data))
    statistics['stdev'] = float(np.std(volume.data))
    statistics['voxelcount'] = volume.data.size
    statistics['shape'] = volume.data.shape
    return statistics


def export_statistics(*volumes: VolumeLike, directory: PathLike) -> None:
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    for volume in volumes:
        filename = build_output_filename_from(volume.fingerprint)
        savepath = directory / filename
        statistics = compute_statistics(volume)

        with savepath.open(mode='w') as handle:
            json.dump(statistics, handle)


if __name__ == '__main__':
    pass