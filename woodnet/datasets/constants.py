"""
Constant mappings about the datasets.

@jsteb 2024
"""
# WOOD_DATA_DIRECTORY = Path(os.environ.get('WOOD_DATA_DIRECTORY'))
from pathlib import Path
from typing import Any

from woodnet.utils import generate_keyset


WOOD_DATA_DIRECTORY = Path('/home/jannik/storage/wood')


COMPLETE_DATASET_DIRECTORY = WOOD_DATA_DIRECTORY / 'complete'
SUBVOLUME_DATASET_DIRECTORY = WOOD_DATA_DIRECTORY / 'chunked'
# default mapping from semantic class names towards the
# numerical values
DEFAULT_CLASSLABEL_MAPPING: dict[Any, int] = {
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