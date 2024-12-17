"""
general datasets implementations.

Jannik stebani 2023
"""

from woodnet.datasets.volumetric import TileDatasetBuilder
from woodnet.datasets.triaxial import TriaxialDatasetBuilder
from woodnet.datasets.planar import (EagerSliceDatasetBuilder,
                                     TiledEagerSliceDatasetBuilder)
from woodnet.datasets.utils import * # noqa: F403
from woodnet.datasets.constants import * # noqa: F403

def get_builder_class(dataset_name: str) -> type:
    """
    Programmatically retrieve the dataset builder class by the
    underlying dataset class name string.
    """
    builder_mapping = {
        'TileDataset' : TileDatasetBuilder,
        'TriaxialDataset' : TriaxialDatasetBuilder,
        'EagerSliceDataset' : EagerSliceDatasetBuilder,
        'TiledEagerSliceDataset' : TiledEagerSliceDatasetBuilder
    }
    return builder_mapping[dataset_name]
