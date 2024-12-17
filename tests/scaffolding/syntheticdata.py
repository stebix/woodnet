"""
Implement the generation of synthetic data.

This yields the data configuration in conjunction with the
zarr dataset creation and materialization inside a test-session
wise temporary directory.

@Author: Jannik Stebani 2024
"""
import uuid
import logging

from pathlib import Path
from collections.abc import Sequence, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import zarr

import pytest

from woodnet.datasets.setup import InstanceFingerprint

PathLike = str | Path
logger = logging.getLogger()

CLASS_NAMES: set[str] = {'acer', 'pinus'}
RAW_DIMENSIONS: tuple[int, int, int] = (1, 320, 1100, 1100)


@dataclass
class ClassSpecification:
    """
    Encapsulates information for a class for the full classification problem.
    """
    name: str
    label: int
    groups: set[str]
    instances_per_group: int
    instances_total: int | None = field(init=False, default=None)

    def __post__init__(self) -> None:
        assert len(self.groups) > 0, 'must specify at least one group'
        self.instances_total = len(self.group) * self.instances_per_group


def create_instance_informations(spec: ClassSpecification,
                                 base_location: Path
                                 ) -> dict[str, dict]:
    """
    Create a number of instance-wise information (ID, location, classname, group)
    from the class-wise specfication. `num` is uitlized to generate ID strings.
    """
    instances = {}
    for group in spec.groups:
        for _ in range(spec.instances_per_group):
            uid = uuid.uuid4() 
            ID = f'test-dataset-{uid}'
            location = base_location / f'{uid}.zarr'

            instances[ID] = {
                'location' : location,
                'classname' : spec.name,
                'group' : group
            }
    return instances


def create_data_configuration(
        specs: Sequence[ClassSpecification],
        base_location: PathLike,
        internal_path: str
    ) -> dict:
    """
    Create a plausible data configuration from a sequence of class specifications.
    """
    base_location = Path(base_location)

    class_to_label_mapping: dict[str, int] = {
        class_spec.name : i
        for i, class_spec in enumerate(specs)
    }

    instance_mapping: dict = {}
    for spec in specs:
        instance_mapping.update(create_instance_informations(spec, base_location))

    data_configuration = {
        'class_to_label_mapping' : class_to_label_mapping,
        'instance_mapping' : instance_mapping,
        'internal_path' : internal_path
    }
    return data_configuration 


def materialize_instance(ID: str,
                         instance: Mapping,
                         raw_dimensions: tuple[int, int, int],
                         internal_path: str) -> Path:
    """
    Materialize an instance from the instance information.
    """
    location = Path(instance['location'])
    classname = instance['classname']
    group = instance['group']

    attrs = {'ID': ID, 'class_': classname, 'group': group}

    array_group_name, array_dataset_name = internal_path.split('/')
    data = np.random.randint(0, 255, size=raw_dimensions, dtype=np.uint8)

    array = zarr.convenience.open(location, mode='w')
    array.attrs.update(attrs)

    array_group = array.require_group(array_group_name)
    array_group.create_dataset(array_dataset_name, data=data)

    logger.info(f'Created synthetic dataset instance at \'{location.resolve()}\'')
    return location

Shape4D = tuple[int, int, int, int]

INTERNAL_PATH: str = 'downsampled/half'
RAW_DIMENSIONS: Shape4D = (1, 320, 320, 320)
BINARY_CLASS_SPEC: list[ClassSpecification] = [
    ClassSpecification(name='acer', label=0, groups={'red', 'green'}, instances_per_group=1),
    ClassSpecification(name='pinus', label=1, groups={'blue', 'green'}, instances_per_group=1),
]



@dataclass
class SyntheticDatasetPackage:
    """
    Joint information and data for a synthetic
    dataset intended for testing purposes.
    """
    internal_path: str
    base_directory: Path
    instance_mapping: dict[str, dict]
    data_configuration: dict[str, Any]


class SyntheticDatasetGenerator:
    default_internal_path: str = INTERNAL_PATH
    default_raw_dimensions: Shape4D = RAW_DIMENSIONS
    default_class_specification: list[ClassSpecification] = BINARY_CLASS_SPEC

    def __init__(self,
                 internal_path: str,
                 raw_dimensions: Shape4D,
                 specs: Sequence[ClassSpecification],
                 cast_fingerprints: bool = True) -> None:
        self.internal_path = internal_path
        self.raw_dimensions = raw_dimensions
        self.specs = specs
        self.cast_fingerprints = cast_fingerprints
    
    @classmethod
    def from_defaults(cls) -> 'SyntheticDatasetGenerator':
        """Generate instance with sensible defaults for binary classification."""
        return cls(internal_path=cls.default_internal_path,
                   raw_dimensions=cls.default_raw_dimensions,
                   specs=cls.default_class_specification)
    

    def generate(self, base_directory: str | Path) -> SyntheticDatasetPackage:
        """Generate the test data on the file system."""
        data_configuration = create_data_configuration(
            self.specs,
            base_directory,
            internal_path=self.internal_path
        )
        instance_mapping: dict = data_configuration['instance_mapping']
        for ID, instance in instance_mapping.items():
            materialize_instance(ID, instance, self.raw_dimensions, self.internal_path)

        if self.cast_fingerprints:
            instance_mapping = {
                k : InstanceFingerprint(**v) for k, v in instance_mapping.items()
            }

        kwargs = {
            'internal_path' : self.internal_path, 'base_directory' : base_directory,
            'instance_mapping' : instance_mapping, 'data_configuration' : data_configuration
        }
        return SyntheticDatasetPackage(**kwargs)




@pytest.fixture(scope='session')
def synthetic_dataset(tmp_path_factory) -> dict[str, Any]:
    """Create a full synthetic dataset in a temporary directory."""
    test_dataset_directory = tmp_path_factory.mktemp('synthetic_datasets')
    generator = SyntheticDatasetGenerator.from_defaults()
    info_bundle = generator.generate(test_dataset_directory)
    return info_bundle


'''
@pytest.fixture(scope='session')
def synthetic_dataset(tmp_path_factory) -> dict[str, Any]:
    """Create a full synthetic dataset in a temporary directory."""
    test_dataset_directory = tmp_path_factory.mktemp('synthetic_datasets')

    INTERNAL_PATH: str = 'downsampled/half'
    BASE_DIRECTORY: str = test_dataset_directory

    specs = [
        ClassSpecification(name='acer', label=0, groups={'red', 'green'}, instances_per_group=1),
        ClassSpecification(name='pinus', label=1, groups={'blue', 'green'}, instances_per_group=1),
    ]
    data_configuration = create_data_configuration(specs, BASE_DIRECTORY, INTERNAL_PATH)
    instance_mapping = data_configuration['instance_mapping']
    for ID, instance in instance_mapping.items():
        _ = materialize_instance(ID, instance, RAW_DIMENSIONS, INTERNAL_PATH)

    info_bundle: dict = {
        'internal_path' : INTERNAL_PATH, 'base_directory' : BASE_DIRECTORY,
        'data_configuration' : data_configuration
    }
    return info_bundle
'''