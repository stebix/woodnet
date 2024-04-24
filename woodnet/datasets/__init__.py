"""
general datasets implementations.

Jannik stebani 2023
"""

from woodnet.datasets.volumetric import TileDatasetBuilder
from woodnet.datasets.triaxial import TriaxialDatasetBuilder
from woodnet.datasets.planar import EagerSliceDatasetBuilder
from woodnet.datasets.utils import *
from woodnet.datasets.constants import *

def get_builder_class(dataset_name: str) -> type:
    """
    Programmatically retrieve the dataset builder class by the
    underlying dataset class name string.
    """
    builder_mapping = {
        'TileDataset' : TileDatasetBuilder,
        'TriaxialDataset' : TriaxialDatasetBuilder,
        'EagerSliceDataset' : EagerSliceDatasetBuilder
    }
    return builder_mapping[dataset_name]
