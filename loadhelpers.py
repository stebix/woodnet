"""
Automated data loading facilities to get datasets directly from their ID.
"""
import os
from pathlib import Path
from itertools import chain

from tqdm.contrib import tqdm_auto
from typing import Iterable, Optional

from datasets import TileDataset
from loader import (LazySlice, SliceLoader, LoadingStrategy,
                    VolumeLoader)

from augmentations import Transformer
from transformbuilder import from_configurations


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




class TileDatasetBuilder:

    internal_path: str = 'downsampled/half'
    classlabel_mapping: dict[str, int] = {
        'ahorn' : 0, 'kiefer' : 1
    }
    # TODO: factor hardcoded paths out -> bad!
    base_directory = '/home/jannik/storage/wood/custom/'

    def build(cls, *IDs: str, phase: str, tileshape: tuple[int],
              transform_configurations: Optional[Iterable[dict]] = None
              ) -> list[TileDataset]:
        datasets = []
        if transform_configurations:
            transformer = Transformer(
                *from_configurations(transform_configurations)
            )
        else:
            transformer = None

        wrapped_IDs = tqdm_auto(IDs, unit='dataset', desc='datasetbuilder')
        for ID in wrapped_IDs:
            wrapped_IDs.set_postfix({'current_ID' : str(ID)})
            path = cls.get_path(ID)        
            dataset = TileDataset(
                path=path, phase=phase, tileshape=tileshape,
                transformer=transformer,
                classlabel_mapping=cls.classlabel_mapping,
                internal_path=cls.internal_path
            )
            datasets.append(dataset)
        return datasets


    @classmethod
    def get_path(cls, ID: str) -> Path:
        base_directory = Path(cls.base_directory)
        for child in base_directory.iterdir():
            if child.match(f'*/{ID}*'):
                return child
        raise FileNotFoundError(f'could not retrieve ID "{ID}" from '
                                f'basedir "{base_directory}"')






