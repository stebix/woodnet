import enum
import logging
import numpy as np

from collections.abc import Iterable, Mapping, Callable
from typing import Literal

import sklearn.model_selection as skml

from woodnet.configtools.validation import TrainingConfiguration
from woodnet.custom.types import PathLike
from woodnet.datasets.constants import CLASS_ID_ORIENTATION_MAPPING



DEFAULT_NAME = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_NAME)


ConfigurationLike = Mapping | TrainingConfiguration


def process(orientation: str) -> str:
    if 'axial' in orientation:
        return 'axiallike'
    elif 'transversal' in orientation:
        return 'transversal'
    raise ValueError(f'invalid orientation \'{orientation}\'')


def convert_to_multiclass(nested: dict[str, dict], /) -> dict:
    """
    Convert a hierarchical nested class to ID to orientation mapping into a
    flat ID to joint woodclass-orientation mapping.

    Example input:
    {'pinus' : {ID1 : 'axial'}, 'acer' : {ID2 : 'transversal'}}

    Example result:
    {ID1 : 'pinus-axial', ID2 : 'acer-transversal'}
    """
    multiclass_mapping = {}
    for class_name, ID_orientation_mapping in nested.items():
        flattened = {
            ID : f'{class_name}-{process(orientation)}'
            for ID, orientation in ID_orientation_mapping.items()
        }
        multiclass_mapping.update(flattened)
    return multiclass_mapping


def split_multiclass_identififers(items: Iterable[str], /) -> tuple[list[str], list[str]]:
    """
    Transform the joint class and orientation identifiers into a clean wood class
    identifier and the corresponding orientation identifier.
    """
    primary_class = []
    secondary_class = []
    for ID in items:
        pcls, scls = ID.split('-')
        primary_class.append(pcls)
        secondary_class.append(scls)
    return (primary_class, secondary_class)


# ordered instance IDs and classes for indexing
INSTANCES, CLASSES = zip(
    *[(ID, class_) for ID, class_ in convert_to_multiclass(CLASS_ID_ORIENTATION_MAPPING).items()]
)
INSTANCES = np.array(INSTANCES)
CLASSES = np.array(CLASSES)
WOOD_CLASSES, ORIENTATION_ClASSES = split_multiclass_identififers(CLASSES)


class CVStrategy(enum.Enum):
    STRATIFIED_KFOLD = 'stratified_kfold'
    STRATIFIED_GROUP_KFOLD = 'stratified_group_kfold'


class _Generator:
    """generates fold-specific set of instance IDs."""
    pass


class StratifiedKFoldsGenerator(_Generator):
    default_n_splits: Literal[3] = 3
    default_random_state: Literal[1701] = 1701
    default_shuffle: bool = True

    def __init__(self,
                 n_splits: int = default_n_splits,
                 random_state: int = default_random_state,
                 shuffle: bool = default_shuffle) -> None:
        super().__init__()
        self._n_splits = n_splits
        self._random_state = random_state
        self._shuffle = shuffle
        
        self._generator = skml.StratifiedKFold(n_splits=self.n_splits,
                                               shuffle=self.shuffle,
                                               random_state=self.random_state)
        self._splits: dict[int, dict] = {}

        foldgenerator = enumerate(self._generator.split(INSTANCES, WOOD_CLASSES), start=1)
        for foldnum, foldspec in foldgenerator:
            training, validation = foldspec
            self._splits[foldnum] = {
                'training' : INSTANCES[training].tolist(),
                'validation' : INSTANCES[validation].tolist()
            }


    def __len__(self) -> int:
        return len(self._splits)


    def __getitem__(self, foldnum: int) -> dict[str, list[str]]:
        try:
            return self._splits[foldnum]
        except KeyError:
            valid = set(self._splits.keys())
            message = (f'Requested invalid fold number {foldnum}. Must be one of {valid}')
            raise KeyError(message)
        

    @property
    def n_splits(self) -> int:
        return self._n_splits
    
    @property
    def random_state(self) -> int:
        return self._random_state
    
    @property
    def shuffle(self) -> int:
        return self._shuffle
    

    def __str__(self) -> str:
        s = (f'{self.__class__.__name__}({self.n_splits=}, {self.random_state=}, '
             f'{self.shuffle=})')
        return s
    
    def __repr__(self) -> str:
        return str(self)
    



class StratifiedGroupKFoldsGenerator(_Generator):
    default_n_splits: Literal[2] = 2
    default_random_state: int | None = None
    default_shuffle: bool = False

    def __init__(self,
                 n_splits: int = default_n_splits,
                 random_state: int = default_random_state,
                 shuffle: bool = default_shuffle) -> None:
        super().__init__()
        self._n_splits = n_splits
        self._random_state = random_state
        self._shuffle = shuffle
        
        self._generator = skml.StratifiedGroupKFold(n_splits=self.n_splits,
                                                    shuffle=self.shuffle,
                                                    random_state=self.random_state)
        self._splits: dict[int, dict] = {}

        foldgenerator = enumerate(self._generator.split(INSTANCES, CLASSES, groups=ORIENTATION_ClASSES), start=1)
        for foldnum, foldspec in foldgenerator:
            training, validation = foldspec
            self._splits[foldnum] = {
                'training' : INSTANCES[training].tolist(),
                'validation' : INSTANCES[validation].tolist()
            }


    def __len__(self) -> int:
        return len(self._splits)


    def __getitem__(self, foldnum: int) -> dict[str, list[str]]:
        try:
            return self._splits[foldnum]
        except KeyError:
            valid = set(self._splits.keys())
            message = (f'Requested invalid fold number {foldnum}. Must be one of {valid}')
            raise KeyError(message)
        

    @property
    def n_splits(self) -> int:
        return self._n_splits
    
    @property
    def random_state(self) -> int:
        return self._random_state
    
    @property
    def shuffle(self) -> int:
        return self._shuffle
    

    def __str__(self) -> str:
        s = (f'{self.__class__.__name__}({self.n_splits=}, {self.random_state=}, '
             f'{self.shuffle=})')
        return s
    
    def __repr__(self) -> str:
        return str(self)



strategy_to_generator: dict[CVStrategy, Callable] = {
    CVStrategy.STRATIFIED_KFOLD : StratifiedKFoldsGenerator,
    CVStrategy.STRATIFIED_GROUP_KFOLD : StratifiedGroupKFoldsGenerator
}


