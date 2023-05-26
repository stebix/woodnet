"""
Automated data loading facilities to get datasets directly from their ID.
"""
import os
from pathlib import Path
from itertools import chain

from typing import Iterable

from loader import (LazySlice, SliceLoader, LoadingStrategy,
                    parse_directory_identifier)

# WOOD_DATA_DIRECTORY = Path(os.environ.get('WOOD_DATA_DIRECTORY'))
WOOD_DATA_DIRECTORY = Path('/home/jannik/storage/wood')

COMPLETE_DATASET_DIRECTORY = WOOD_DATA_DIRECTORY / 'complete'
SUBVOLUME_DATASET_DIRECTORY = WOOD_DATA_DIRECTORY / 'chunked'

EXPECTED_IDS = {'CT20', 'CT3', 'CT5', 'CT7', 'CT6', 'CT8', 'CT9',
                'CT21', 'CT22', 'CT16', 'CT17', 'CT2', 'CT14',
                'CT11', 'CT19', 'CT18', 'CT10', 'CT12', 'CT13'}

class_ID_orientation: dict[str, dict] = {
    'ahorn' : {
        'CT16' : 'axial-tangential',
        'CT17' : 'axial-tangential',
        'CT2' : 'transversal',
        'CT14' : 'transversal',
        'CT11' : 'transversal',
        'CT19' : 'axial-tangential',
        'CT18' : 'axial-tangential',
        'CT10' : 'transversal',
        'CT12' : 'transversal',
        'CT13' : 'transversal'
    },
    'kiefer' : {
        'CT20' : 'axial',
        'CT3' : 'transversal',
        'CT5' : 'transversal',
        'CT7' : 'transversal',
        'CT6' : 'transversal',
        'CT8' : 'transversal',
        'CT9' : 'transversal',
        'CT21' : 'axial',
        'CT22' : 'axial'
    }
}

loader = SliceLoader()
loader.strategy = LoadingStrategy.LAZY


if not COMPLETE_DATASET_DIRECTORY.is_dir():
    raise FileNotFoundError(
        f'Complete directory not found at location inferred from '
        f'environment: "{COMPLETE_DATASET_DIRECTORY}"'
    )

if not SUBVOLUME_DATASET_DIRECTORY.is_dir():
    raise FileNotFoundError(
        f'Subvolumes directory not found at location inferred from '
        f'environment: "{SUBVOLUME_DATASET_DIRECTORY}"'
    )


def retrieve_directory(ID: str) -> Path:
    for child in COMPLETE_DATASET_DIRECTORY.iterdir():
        if not child.is_dir():
            continue
        if child.match(f'{ID}_*'):
            return child
    raise FileNotFoundError(f'ID {ID} directory not found @ expected '
                            f'location: {COMPLETE_DATASET_DIRECTORY}')


def instance(ID: str) -> list[LazySlice]:
    directory = retrieve_directory(ID)
    return loader.from_directory(directory)


def instances(IDs: Iterable[str]) -> list[LazySlice]:
    slices = [instance(ID) for ID in IDs]
    return list(chain.from_iterable(slices))


def get_ID_by(class_: str, orientation: None | str = None) -> list[str]:
    IDs_mapping = class_ID_orientation[class_]
    if orientation:
        IDs = [
            ID for ID, ostate in IDs_mapping.items() if ostate == orientation
        ]
    else:
        IDs = list(IDs_mapping.keys())
    return IDs


def subvolume(ID: str, index: int) -> list[LazySlice]:
    raise NotImplementedError('not yet son')













