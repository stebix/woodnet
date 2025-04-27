import logging
from pathlib import Path
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Literal

import numpy as np
import attrs
import zarr
import zarr.errors

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger()

PathLike = str | Path 

def scrape_directory(
    path: PathLike
) -> dict[str, Path]:
    scraped_zarr_items: dict[str, Path] = {}
    path = Path(path) if not isinstance(path, Path) else path
    for item in path.iterdir():
        if not item.is_dir() or not item.name.endswith('.zarr'):
            logging.info(f'Skipping {item}')
            continue
        scraped_zarr_items[item.stem] = item
    return scraped_zarr_items




def get_zarr_structure(path: PathLike) -> dict[str, str]:
    """
    Extracts the structural hierarchy of a Zarr store.

    Parameters
    ----------
    zarr_path: Path | str
        Location of the Zarr store (directory or zip).

    Returns
    -------
        A dictionary mapping full item paths to their type ('group' or 'array').
    """
    structure = {}
    try:
        root = zarr.open_group(path, mode='r')

        # Handle the root group itself: Zarr root is always a group
        structure['/'] = 'group'

        def _visitor(name, obj):
            # visititems gives path relative to root, ensure leading '/'
            full_path = f'/{name}'
            if isinstance(obj, zarr.Group):
                structure[full_path] = 'group'
            elif isinstance(obj, zarr.Array):
                structure[full_path] = 'array'
            else:
                raise TypeError(f'Unknown beast: Zarr object type: {type(obj)}')

        root.visititems(_visitor)

    except zarr.errors._BaseZarrError as e:
        msg = f'Failed to parse Zarr structure at location \'{path}\''
        raise RuntimeError(msg) from e

    return structure



def diff_structures(
    struct1: dict[str, str],
    struct2: dict[str, str],
    name1: str = "Object 1",
    name2: str = "Object 2"
) -> list[str]:
    """
    Compares two Zarr structure dictionaries and generates a diff report.

    Parameters
    ----------
    struct1: dict[str, str]
        Structure dictionary of the first object.

    struct2: dict[str, str]
        Structure dictionary of the second object.
    
    name1: str, optional
        Name for the first object.
        
    name2: str, optional
        Name for the second object.

    Returns
    -------
        A list of strings representing the structural diff.
    """
    paths1 = set(struct1.keys())
    paths2 = set(struct2.keys())

    removed_paths = sorted(list(paths1 - paths2))
    added_paths = sorted(list(paths2 - paths1))
    common_paths = sorted(list(paths1 & paths2))

    diff_lines = []
    diff_lines.append(f"--- {name1}")
    diff_lines.append(f"+++ {name2}")

    for path in removed_paths:
        # Ensure path exists in struct1 (should always be true here)
        if path in struct1:
             diff_lines.append(f"- [{struct1[path]}] {path}")

    for path in added_paths:
        # Ensure path exists in struct2 (should always be true here)
        if path in struct2:
            diff_lines.append(f"+ [{struct2[path]}] {path}")

    for path in common_paths:
        type1 = struct1.get(path)
        type2 = struct2.get(path)
        if type1 != type2:
            diff_lines.append(f"M [{type1} -> {type2}] {path}")
        # else:
            # Optionally add unchanged lines:
            # diff_lines.append(f"  [{type1}] {path}")

    # Add a summary line if useful
    if not removed_paths and not added_paths and not any(struct1.get(p) != struct2.get(p) for p in common_paths):
         diff_lines.append("No structural differences found.")
    elif not diff_lines[2:]: # Only headers added
         diff_lines.append("No structural differences found.")


    return diff_lines


@attrs.define
class ZarrArrayProxy:
    """
    Represents a proxy for a Zarr array, providing metadata and shape information.
    """
    metadata: Mapping
    shape: tuple[int, ...]
    inpath: PathLike
    outpath: PathLike

    def load(self) -> np.ndarray:
        """"Load the array data from the Zarr store."""
        zarr_array = zarr.convenience.open(self.outpath, mode='r')[self.inpath]
        return zarr_array[...]


def sanitize_name(s: str) -> str:
    return s.replace('-', '_')


def _require_group_hierarchy(
    parent: SimpleNamespace,
    groupnames: Sequence[str],
) -> SimpleNamespace:
    """
    Ensure that the group hierarchy exists in the parent object.

    Parameters
    ----------
    parent : SimpleNamespace
        The parent object to which the group hierarchy will be added.
    
    groupnames : Sequence[str]
        A sequence of group names representing the hierarchy to be created.
        Simple sequential 'depth-first' specification.

    Returns
    -------
        The leaf namespace object of the group hierarchy.
    """
    for subgroup_name in groupnames:
        if hasattr(parent, subgroup_name):
            # group is present, just go deeper
            parent = getattr(parent, subgroup_name)
        else:
            # group is not present, create it
            # and go deeper
            subgroup = SimpleNamespace()
            setattr(parent, subgroup_name, subgroup)
            parent = subgroup
    return parent


def make_proxy(path: PathLike) -> SimpleNamespace:
    """
    Create a proxy object for a Zarr store for easy inspection of
    structure and array content metadata.

    Parameters
    ----------
    path : PathLike
        Path to the Zarr store.
    
    Returns
    -------
        A SimpleNamespace object reclecting the internal (nested)
        Zarr store structure.
        Leaf nodes are ZarrArrayProxy objects with metadata and shape.
    """
    MODE: Literal['r'] = 'r'
    structure = get_zarr_structure(path)
    root = SimpleNamespace()
    for key, value in structure.items():
        if key == '/':
            # root is already included
            continue
        if value == 'group':
            parent = root
            names = [sanitize_name(name) for name in key.split('/') if name != '']
            parent = _require_group_hierarchy(parent, names)
                
        elif value == 'array':
            parent = root
            names = [sanitize_name(name) for name in key.split('/')[:-1] if name != '']
            array_name = sanitize_name(key.split('/')[-1])
            parent = _require_group_hierarchy(parent, names)
            array = zarr.convenience.open(path, mode=MODE)[key]
            array_shape = array.shape
            array_metadata = dict(array.attrs)
            proxy = ZarrArrayProxy(
                metadata=array_metadata, shape=array_shape,
                inpath=key, outpath=path
            )
            setattr(parent, array_name, proxy)
                
    return root
        