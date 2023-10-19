"""
general datasets implementations.

Jannik stebani 2023
"""
import numpy as np

from itertools import chain
from pathlib import Path

from woodnet.utils import generate_keyset

# WOOD_DATA_DIRECTORY = Path(os.environ.get('WOOD_DATA_DIRECTORY'))
WOOD_DATA_DIRECTORY = Path('/home/jannik/storage/wood')

COMPLETE_DATASET_DIRECTORY = WOOD_DATA_DIRECTORY / 'complete'
SUBVOLUME_DATASET_DIRECTORY = WOOD_DATA_DIRECTORY / 'chunked'

# default mapping from semantic class names towards the
# numerical values
DEFAULT_CLASSLABEL_MAPPING = {
    'acer' : 0,
    'pinus' :  1
}

CLASSNAME_REMAP: dict[str, str] = {
    'ahorn' : 'acer',
    'kiefer' : 'pinus'
}

# single source of truth for dataset ID class and orientation
CLASS_ID_ORIENTATION_MAPPING: dict[str, dict] = {
    'acer' : {
        'CT16' : 'axial-tangential',
        'CT17' : 'axial-tangential',
        'CT19' : 'axial-tangential',
        'CT18' : 'axial-tangential',
        'CT14' : 'transversal',
        'CT11' : 'transversal',
        'CT2' : 'transversal',
        'CT10' : 'transversal',
        'CT12' : 'transversal',
        'CT13' : 'transversal'
    },
    'pinus' : {
        'CT3' : 'transversal',
        'CT5' : 'transversal',
        'CT7' : 'transversal',
        'CT6' : 'transversal',
        'CT8' : 'transversal',
        'CT9' : 'transversal',
        'CT15' : 'axial',
        'CT20' : 'axial',
        'CT21' : 'axial',
        'CT22' : 'axial'
    }
}

VALID_IDS: set[str] = generate_keyset(CLASS_ID_ORIENTATION_MAPPING.values())


def add_channel_dim(array: np.ndarray) -> np.ndarray:
    """Add fake channel dimension."""
    return array[np.newaxis, ...]


def get_spatial_shape(shape: tuple[int]) -> tuple[int]:
    """Get spatial shape for 4D inputs"""
    return shape[1:]



def retrieve_directory(ID: str) -> Path:
    for child in COMPLETE_DATASET_DIRECTORY.iterdir():
        if not child.is_dir():
            continue
        if child.match(f'{ID}_*'):
            return child
    raise FileNotFoundError(f'ID {ID} directory not found @ expected '
                            f'location: {COMPLETE_DATASET_DIRECTORY}')


def get_ID_by(class_: str, orientation: None | str = None) -> list[str]:
    IDs_mapping = CLASS_ID_ORIENTATION_MAPPING[class_]
    if orientation:
        IDs = [
            ID for ID, ostate in IDs_mapping.items() if ostate == orientation
        ]
    else:
        IDs = list(IDs_mapping.keys())
    return IDs

