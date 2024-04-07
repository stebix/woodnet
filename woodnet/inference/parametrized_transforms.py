"""
Facilities to create and handle parametrized data transforms to
perform data ablation experiments.

@jsteb 2024
"""
import os
import logging
from collections import UserList
from collections.abc import Mapping, Callable, Sequence, Iterable
from typing import NamedTuple, Any, Type

import tqdm


LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)

def deduce_display_progressbar() -> bool:
    env_value = os.environ.get('DISPLAY_PROGRESSBAR', None)
    logger.debug(f'retrieved environment value: DISPLAY_PROGRESSBAR = \'{env_value}\'')
    falsy: set[str] = {'0', 'false', 'False'}
    return env_value not in falsy


# make progress reporting steerable via environment variables
DISPLAY_PROGRESSBAR: bool = deduce_display_progressbar()


class ParametrizedTransform(NamedTuple):
    """
    Simple struct to hold human-readable string name, parameter settings
    and transform itself.
    """
    name: str
    parameters: Mapping
    transform: Callable


def maybe_wrap_parametrized_transforms(transforms: Sequence[Callable],
                                       name: str = 'notset',
                                       display_progress: bool = DISPLAY_PROGRESSBAR
                                       ) -> Sequence[Callable]:
    """
    Conditionally wrap a sequence of data transformations with a
    tqdm progress bar.

    Parameters
    ----------

    name : str
        Name of the transformation. Will be displayed in the progress bar.

    transforms : sequence of Callable
        Data transformations.

    display_progress : bool, optional
        Boolean switch for progress visualization.
        Defaults
    """
    if not display_progress:
        return transforms
    settings = {'unit' : 'stage', 'desc' : name, 'leave' : False}
    return tqdm.tqdm(transforms, **settings)


def extract_class_name_with_namespace(class_or_instance: type | Any) -> str:
    """Extract class name with definite namespace prefix, e.g. 'app.module.FooClass'"""
    if not isinstance(class_or_instance, type):
        class_or_instance = type(class_or_instance)
    parts = str(class_or_instance).split("'")
    # usually parts is formed like: ['<class ', 'app.mymodule.FooClass', '>']
    # from which we want to take the class name with the namespace prefix
    return parts[1]


class CoherentList(UserList):
    """
    Container for coherent parametrized transforms.
    """
    def __init__(self, iterable: Iterable[ParametrizedTransform] | None = None) -> None:
        super().__init__()
        self._name: str = ''
        self._parameter_set: set[str] = set()
        self._transform_class: Type | None = None
        iterable = iterable or []
        for item in iterable:
            self.append(item)
    
    def __setitem__(self, index: int, item: ParametrizedTransform) -> None:
        item = self._validate(item)
        self.data[index] = item
        
    def insert(self, index: int, item: ParametrizedTransform) -> None:
        self.data.insert(index, self._validate(item))
        
    def append(self, item: ParametrizedTransform) -> None:
        self.data.append(self._validate(item))
    
    def extend(self, other: Iterable | 'CoherentList') -> None:
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super.extend(self._validate(item) for item in other)
    
    def _validate(self, item: ParametrizedTransform):
        if len(self) == 0:
            self._name = item.name
            self._parameter_set = set(item.parameters.keys())
            self._transform_class = type(item.transform)
            return item
        
        if (item.name == self._name
                and item.parameters.keys() == self._parameter_set
                and type(item.transform) == self._transform_class
           ):
            return item
        else:
            raise ValueError(f'attempting to add non-conforming item: {item}')
            
    def info(self) -> dict:
        # use extract_class_name_with_namespace to visualize possible class definition shadowing
        # when using different modules from which similarly named transforms are imported
        info = {'name' : self._name,
                'parameter_set' : self._parameter_set,
                'transform_class' : extract_class_name_with_namespace(self._transform_class)}
        return info


