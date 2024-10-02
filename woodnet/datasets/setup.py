"""
Central location for entire data setup.

@Jannik Stebani 2024
"""
from pathlib import Path
from typing import Any
from pydantic import BaseModel, field_validator, ValidationError

from woodnet.custom.exceptions import ConfigurationError
from woodnet.configtools import load_yaml
from woodnet.utils import generate_keyset


PathLike = Path | str

# TODO: factor out hardcoded path -> move to ENV or CONF
DATA_CONFIGURATION_PATH: PathLike = Path('/home/jannik/code/woodnet/tests/_dataconf.yaml')

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



def group_instances_by_class(instance_mapping: dict[str, InstanceFingerprint]) -> dict[str, list[str]]:
    """
    Group instances by class name.
    """
    class_to_instances: dict[str, str] = {}
    for ID, fingerprint in instance_mapping.items():
        if fingerprint.classname not in class_to_instances:
            class_to_instances[fingerprint.classname] = []
        class_to_instances[fingerprint.classname].append(ID)
    return class_to_instances
    

DATA_CONFIGURATION = load_data_configuration(DATA_CONFIGURATION_PATH)
INSTANCE_MAPPING = DATA_CONFIGURATION.instance_mapping
INTERNAL_PATH = DATA_CONFIGURATION.internal_path
CLASSLABEL_MAPPING = DATA_CONFIGURATION.class_to_label_mapping