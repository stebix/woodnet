"""
Facilities to create and handle parametrized data transforms to
perform data ablation experiments.

@jsteb 2024
"""
import os
import logging
import importlib
import torch

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
    transform: Callable[[torch.Tensor], torch.Tensor]



class CongruentTransformList(UserList):
    """
    Managed list-like container for parametrized transforms that guarantees
    that only congruent parametrized transforms are held inside.
    Congruent means:
        - similar name (should be semantically meaningful for the intended transform)
        - similar transform class
        - similar parameter sets, but parameter values may differ (-> sequence of transforms)

    Parameters
    ----------

    iterable : Iterable of ParametrizedTransform or None
        Initial data for the managed container.
        None means empty container. Defaults to None.

    
    Attributes
    ----------

    name : str
        Global semantic name of the parametrized transform.

    parameter_set : set of str
        Global congruent parameter set of the transforms.

    transform_class_identifier : str
        Class name with namespace prefix of the parametrized transform.

    transform_class : type
        Class object of the parametrized transform.
   

        
    Methods
    -------

    info()
        Return a information dictionary about the contained
        parametrized transforms.
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
    
    def extend(self, other: Iterable) -> None:
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
        info = {'name' : self.name,
                'parameter_set' : self.parameter_set,
                'transform_class' : self.transform_class_identifier}
        return info
    
    @property
    def name(self) -> str:
        """Global name of the parametrized transform instances."""
        return self._name
    
    @property
    def parameter_set(self) -> set[str]:
        """Global parameter set of the parametrized transform instances."""
        return self._parameter_set
    
    @property
    def transform_class_identifier(self) -> str:
        """Global transform class with namespace of the parametrized transform instances."""
        return extract_class_name_with_namespace(self._transform_class)

    @property
    def transform_class(self) -> type:
        """Global transform class object of the parametrized transform instances."""
        return self._transform_class


# Type aliases for sequences of parametrized transforms
Parametrizations = list[ParametrizedTransform]
CongruentParametrizations = CongruentTransformList[ParametrizedTransform]


"""
General helper functions and instantiation tooling.
"""


def get_transform_class(name: str) -> Type:
    """
    Programmatically retrieve the transform class from cusotm code or
    monai packagge.
    """
    module_names = ('monai.transforms', 'woodnet.transformations.transforms')
    for module_name in module_names:
        module = importlib.import_module(module_name)
        try:
            class_ = getattr(module, name)
            break
        except AttributeError:
            pass
    else:
        # we did not find the class via string name in any of the modules
        raise AttributeError(f'could not retrieve requested transform class \'{name}\''
                             f'from modules {module_names}')
    
    logger.debug(f'retrieved class object {class_} for module \'{module_name}\'')
    return class_


def _generate_parametrized_transforms(specification: Mapping) -> list[ParametrizedTransform]:
    """
    Programmatically generate multiple parametrized transforms from a single `specification` mapping.

    Parameters
    ----------

    specification : Mapping
    """
    name = specification.get('name')
    class_name = specification.get('class_name', None) or name
    parameters = specification.get('parameters')
    transform_class = get_transform_class(class_name)
    transforms = []
    for combination in parameters:
        transform = transform_class(**combination)
        transforms.append(
            ParametrizedTransform(name=name, parameters=combination, transform=transform)
        )
    return transforms


def generate_parametrized_transforms(*specifications: Mapping,
                                     squeeze: bool = True
                                     ) -> list[Parametrizations] | Parametrizations:
    """
    Programmatically generate parametrized transforms from an abitrary number
    of `specifcation` mappings.
    """
    transforms = [_generate_parametrized_transforms(spec) for spec in specifications]
    selector = 0 if len(transforms) == 1 and squeeze else slice(None)
    return transforms[selector]


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


