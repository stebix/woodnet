"""
Central location for entire data setup.

@Jannik Stebani 2024
"""
import os
import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal, NamedTuple
from pydantic import BaseModel, field_validator, ValidationError

from woodnet.custom.exceptions import ConfigurationError
from woodnet.configtools import load_yaml


PathLike = Path | str

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def load_env_file(filepath: Path) -> dict:
    """Load the content of a simple key-value .env file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        content = {
            k : v for k, v in map(lambda x: x.split('='), lines)
        }
    return content


def retrieve_data_configuration_path() -> Path:
    """
    Retrieve a tentative data configuration path from user sepcification.
    The path can originate from an environment variable or specified
    in the .env file at the repository root.

    We can still fail later if the specified path does not exist or is not a file.

    TODO:   this is brittle: store in configuration in user home directory.
            But then we have to pay attention to the platform we are running on.
    """
    file_path = Path(__file__)
    # go up -> datasets -> woodnet -> src -> repository root
    tentative_env_file_path = file_path.parents[3] / '.env'

    if tentative_env_file_path.exists():
        env_file = load_env_file(tentative_env_file_path)
        dataconf_path_envfile = env_file.get('DATA_CONFIGURATION_PATH', None)        
    else:
        logger.info(f'Env file not found at expected location: \'{tentative_env_file_path}\'')

    dataconf_path_environ = os.environ.get('DATA_CONFIGURATION_PATH', None)

    both = dataconf_path_envfile and dataconf_path_environ
    by_envvar = dataconf_path_environ and dataconf_path_envfile is None
    by_envfile = dataconf_path_environ is None and dataconf_path_envfile

    if both:
        # we got both info, but environment variable has higher precedence
        logger.info(
            f'Retrieved data configuration paths from both environment variable '
            f'and .env file. Using environment variable path due to higher precedence'
        )
        dataconf_path = Path(dataconf_path_environ)
    elif by_envvar:
        # we only have the environment variable
        logger.info('Retrieved data configuration path from environment variable.')
        dataconf_path = Path(dataconf_path_environ)
    elif by_envfile:
        # we only have the .env file
        logger.info('Retrieved data configuration path from .env file.')
        dataconf_path = Path(dataconf_path_envfile)
    else:
        # we have no information where the data configuration is
        raise ConfigurationError(
            'No data configuration path found. Please specify in .env file or as environment variable.'
        )
    return dataconf_path


DATA_CONFIGURATION_PATH: PathLike = retrieve_data_configuration_path()


class InstanceFingerprint(BaseModel):
    location: str
    classname: str
    group: str | None

    @field_validator('location')
    def location(cls, value: str) -> Path:
        return Path(value)


class DataConfiguration(BaseModel):
    """
    Joint configuration for the data setup.
    Here all information about the data location and configuration is stored.
    """
    internal_path: str
    class_to_label_mapping: dict[str, int]
    instance_mapping: dict[str, InstanceFingerprint]


def load_data_configuration(filepath: Path) -> DataConfiguration:
    """
    Load the data configuration from a yaml file.

    Parameters
    ----------

    filepath : Path
        Path to the yaml file that is the data configuration.
        It should contain the class to label mapping and the instance mapping.
    """
    raw_configuration: dict = load_yaml(filepath)
    try:
        data_configuration = DataConfiguration(**raw_configuration)
    except ValidationError as e:
        raise ConfigurationError(
            f'Error in data configuration file \'{filepath}\''
        ) from e
    return data_configuration



def _group_by_class_detailed(instance_mapping: dict[str, InstanceFingerprint]
                             ) -> dict[str, dict[str, InstanceFingerprint]]:
    class_to_instances = defaultdict(dict)
    for ID, fingerprint in instance_mapping.items():
        class_to_instances[fingerprint.classname][ID] = fingerprint    
    return class_to_instances


def _group_by_class_concise(instance_mapping: dict[str, InstanceFingerprint]
                            ) -> dict[str, list[str]]:
    class_to_instances = defaultdict(list)
    for ID, fingerprint in instance_mapping.items():
        class_to_instances[fingerprint.classname].append(ID)
    return class_to_instances


def group_instances_by_class(instance_mapping: dict[str, InstanceFingerprint],
                             format: Literal['list', 'mapping'] = 'list'
                             ) -> dict[str, list[str]]:
    """
    Group instances by class name.

    Parameters
    ----------

    instance_mapping : dict[str, InstanceFingerprint]
        Mapping of instance IDs to instance fingerprints.

    format : Literal['list', 'mapping']
        Format of the output.
        If 'list', the output will be a concise list of instance IDs for each class.
        If 'mapping' the output will be a dictionary with instance IDs as keys
        and remaining fingerprint data as dictionay values.

    Returns
    -------

    class_to_instances : dict[str, list[str]] | dict[str, dict[str, InstanceFingerprint]]
    """
    if format == 'list':
        return _group_by_class_concise(instance_mapping)
    elif format == 'mapping':
        return _group_by_class_detailed(instance_mapping)
    else:
        raise ValueError(f'Invalid format \'{format}\'. Use \'list\' or \'mapping\'.')
    

DATA_CONFIGURATION = load_data_configuration(DATA_CONFIGURATION_PATH)
#
INSTANCE_MAPPING = DATA_CONFIGURATION.instance_mapping
INTERNAL_PATH = DATA_CONFIGURATION.internal_path
CLASSLABEL_MAPPING = DATA_CONFIGURATION.class_to_label_mapping


class InstanceMappingLists(NamedTuple):
    IDs: list[str]
    CLASSES: list[str]
    GROUPS: list[str]


def convert_to_lists(instance_mapping: dict[str, InstanceFingerprint]) -> dict[str, list[str]]:
    """Convert the nested mapping to three flat lists
    
    The instance mapping maps instance IDs to instance fingerprints, which in turn
    possess information about the data location on the filesystem, the class name
    and the group of the instance. This function converts the nested mapping to three lists
    where the information at an index is related to the same instance.
    """
    IDS = []
    CLASSES = []
    GROUPS = []

    for ID, fingerprint in instance_mapping.items():
        IDS.append(ID)
        CLASSES.append(fingerprint.classname)
        GROUPS.append(fingerprint.group)

    return InstanceMappingLists(IDS, CLASSES, GROUPS)
