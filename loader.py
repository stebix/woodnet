from pathlib import Path
import numpy as np
from typing import Optional

from customtypes import PathLike
from dataobjects import SubvolumeFingerprint, EagerSlice



def parse_directory_identifier(identifier: str) -> dict:
    """Parse wood project identifier scheme strings for CT data."""
    # clumsy but ok
    (ID, class_, voltage,
     current, duration, averages) = identifier.split('_')
    attributes = {
        'ID' : ID, 'class_' : class_, 'voltage' : voltage,
        'current' : current, 'duration' : duration,
        'averages' : averages
    }
    return attributes



def parse_filename_identifier(identifier: str) -> dict:
    (ID, class_, voltage, current,
     duration, averages, last) = identifier.split('_')
    index, _ = last.split('.')
    attributes = {
        'ID' : ID, 'class_' : class_, 'voltage' : voltage,
        'current' : current, 'duration' : duration,
        'averages' : averages, 'index' : index
    }
    return attributes



def postprocess_ID(ID: str) -> str:
    return int(ID.removeprefix('CT'))

def postprocess_class(class_: str) -> str:
    return class_.lower()

def postprocess_averages(averages: str) -> str:
    return int(averages.removesuffix('mitt'))

def postprocess_slice_index(slice_index: str) -> int:
    return int(slice_index)


def postprocess(attributes: dict) -> dict:
    identity = lambda arg: arg
    func_mapping = {
        'ID' : postprocess_ID,
        'class_' : postprocess_class,
        'averages' : postprocess_averages,
        'slice_index' : postprocess_slice_index
    }
    print(f'received attributes :: {attributes}')
    postprocessed = {
        key : func_mapping.get(key, identity)(value)
        for key, value in attributes.items()
    }
    return postprocessed



class SubvolumeLoader:
    """
    Load a subvolume as a monlithic `numpy.ndarray` object.
    """
    suffix: str = 'tif'
    subvolume_directory_prefix: str = 'subvol'

    def from_directory(self, directory: PathLike) -> np.ndarray:
        directory = Path(directory)
        level, attributes = self.deduce_directory_level(directory)
        if level == 'top':
            if index is None:
                raise ValueError(
                    f'For given top-level directory "{directory}" a subvolume index '
                    f'is required for disambiguation, but got None'
                )
            directory = directory / '_'.join((self.subvolume_directory_prefix, str(index)))
            attributes['index'] = index
        else:
            pass

        paths = [
            element for element in directory.iterdir()
            if element.name.endswith(self.suffix)
        ]


    def from_top_directory(self, directory: PathLike, index: int) -> np.ndarray:
        directory = Path(directory)
        attributes = postprocess(parse_directory_identifier(directory.stem))
        attributes['index'] = index

        directory = directory / '_'.join((self.subvolume_directory_prefix, str(index)))
        fingerprint = SubvolumeFingerprint(**attributes)
        slices = []
        for element in directory.iterdir():
            if not element.name.endswith(self.suffix):
                continue
            attributes = parse_filename_identifier(element.name)
            index = attributes.pop('index')
            slices.append(
                EagerSlice(filepath=element, fingerprint=fingerprint,
                           index=index)
            )
        # sort by slice index
        slices = sorted(slices, key=lambda s: s.index)
        return slices













