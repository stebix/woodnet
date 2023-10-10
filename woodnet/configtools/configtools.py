import pathlib
from collections.abc import Mapping

from ruamel.yaml import YAML

from woodnet.custom.types import PathLike

def load_yaml(path: PathLike) -> Mapping:
    """Load content of th YAML file from the indicated location."""
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    yaml = YAML()
    with path.open(mode='r') as handle:
        content = yaml.load(handle)

    return content