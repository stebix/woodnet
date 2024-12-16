import numpy as np
import uuid
import logging

from pathlib import Path
from collections.abc import Sequence, Mapping
from dataclasses import dataclass, field

import zarr

import pytest
import zarr.convenience


PathLike = str | Path
logger = logging.getLogger()

CLASS_NAMES: set[str] = {'acer', 'pinus'}
RAW_DIMENSIONS: tuple[int, int, int] = (320, 1100, 1100)


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

    logger.info(f'Create synthetic dataset instance at \'{location.resolve()}\'')
    return location



@pytest.fixture(scope='session')
def synthetic_dataset(tmp_path_factory) -> dict:
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
