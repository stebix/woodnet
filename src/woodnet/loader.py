import enum
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Iterable

from woodnet.dataobjects import (SubvolumeFingerprint, Subvolume, AbstractSlice,
                                 CachingSlice, LazySlice, InstanceFingerprint, Volume)
from woodnet.custom.types import PathLike


DEFAULT_SUFFIX: str = 'tif'
DEFAULT_SUBVOLUME_PREFIX: str = 'subvol'


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
    index, suffix = last.split('.')
    attributes = {
        'ID' : ID, 'class_' : class_, 'voltage' : voltage,
        'current' : current, 'duration' : duration,
        'averages' : averages, 'index' : index,
        'suffix' : suffix
    }
    return attributes



def postprocess_ID(ID: str) -> str:
    return int(ID.removeprefix('CT'))

def postprocess_class(class_: str) -> str:
    return class_.lower()

def postprocess_averages(averages: str) -> str:
    return int(averages.removesuffix('mitt'))

def postprocess_index(index: str) -> int:
    return int(index)


def postprocess(attributes: dict) -> dict:
    identity = lambda arg: arg
    func_mapping = {
        'ID' : postprocess_ID,
        'class_' : postprocess_class,
        'averages' : postprocess_averages,
        'index' : postprocess_index
    }
    postprocessed = {
        key : func_mapping.get(key, identity)(value)
        for key, value in attributes.items()
    }
    return postprocessed



class LoadingStrategy(enum.Enum):
    LAZY = enum.auto()
    CACHING = enum.auto()


def is_subvolume_directory(directory: Path) -> bool:
    """
    Estimate on last path component whether the given directory containes
    slices of a subvolume.
    """
    if directory.stem.startswith(DEFAULT_SUBVOLUME_PREFIX):
        return True
    return False



def fingerprint_from_directory(directory: PathLike) -> InstanceFingerprint:
    directory = Path(directory)
    if is_subvolume_directory(directory):
        base_directory = directory.parent
        subvolume_directory = directory.stem
        attributes = postprocess(
            parse_directory_identifier(base_directory.stem)
        )
        index = int(subvolume_directory.split('_')[-1])
        attributes['index'] = index
        fingerprint = SubvolumeFingerprint(**attributes)
    else:
        attributes = postprocess(
            parse_directory_identifier(directory.stem)
        )
        fingerprint = InstanceFingerprint(**attributes)
    return fingerprint


def fingerprint_from_filepath(filepath: PathLike) -> InstanceFingerprint:
    filepath = Path(filepath)





class SliceLoader:
    """
    Load 2D slices from a directory. 
    """
    suffix: str = DEFAULT_SUFFIX
    recursive: bool = False
    strategy: LoadingStrategy = LoadingStrategy.CACHING

    def from_directory(self, directory: Path) -> list[AbstractSlice]:
        if self.strategy is LoadingStrategy.CACHING:
            slice_class = CachingSlice
        else:
            slice_class = LazySlice

        slices = []
        fingerprint = fingerprint_from_directory(directory)
        for item in directory.iterdir():
            try:
                fileattributes = postprocess(
                    parse_filename_identifier(item.name)
                )
            except (SyntaxError, AttributeError, ValueError) as e:
                warnings.warn(f'Malformed slice candidate: "{item}". '
                              f'Could not parse due to: {e}')
                continue
            if fileattributes['suffix'] != DEFAULT_SUFFIX:
                continue
            slices.append(
                slice_class(filepath=item, fingerprint=fingerprint,
                            index=fileattributes['index'])
            )
        return slices



def get_progressbar_wrapper(progressbar: str) -> callable:
    """Get progressbar depending on selection."""
    if progressbar == 'tqdm':
        from tqdm import tqdm
        return tqdm
    elif progressbar =='tqdm_notebook':
        from tqdm.notebook import tqdm_notebook as tqdm
        return tqdm
    elif progressbar == 'tqdm_auto':
        from tqdm.auto import tqdm
        return tqdm
    elif progressbar == 'none':
        return lambda x: x
    else:
        raise ValueError(f'requested invalid progressbar wrapper: '
                         f'"{progressbar}"')


def maybe_wrap_progressbar(iterable: Iterable, progressbar: str,
                           kwargs: dict) -> Iterable:
    wrapper = get_progressbar_wrapper(progressbar)
    return wrapper(iterable, **kwargs)



class VolumeLoader:
    """
    Load all slices from a directory as a monolithic `numpy.ndarray` volume.
    """
    suffix: str = DEFAULT_SUFFIX
    progressbar: str = 'tqdm_auto'
    progressbar_kwargs: dict = {}
    
    def from_directory(self, directory: PathLike) -> Volume:
        directory = Path(directory)
        attributes = postprocess(parse_directory_identifier(directory.stem))
        fingerprint = InstanceFingerprint(**attributes)
        tifs = [
            f for f in directory.iterdir()
            if f.is_file() and f.name.endswith(self.suffix)
        ]
        slices = []
        for f in tifs:
            file_attributes = parse_filename_identifier(f.name)
            index = int(file_attributes['index'])
            slices.append(
                CachingSlice(filepath=f, fingerprint=fingerprint, index=index)
            )
        slices = sorted(slices, key=lambda s: s.index)
        # actual loading is performed in stack operation
        slices = maybe_wrap_progressbar(slices, self.progressbar,
                                        self.progressbar_kwargs)
        volume = np.stack(tuple(s.data for s in slices))
        return Volume(directorypath=directory, fingerprint=fingerprint, data=volume)





class SubvolumeLoader:
    """
    Load a subvolume as a monolithic `numpy.ndarray` object.
    """
    suffix: str = DEFAULT_SUFFIX
    subvolume_prefix: str = DEFAULT_SUBVOLUME_PREFIX


    def _from_directory(self, directory: Path, attributes: dict) -> Subvolume:
        fingerprint = SubvolumeFingerprint(**attributes)
        slices = []
        for element in directory.iterdir():
            if not element.name.endswith(self.suffix):
                continue
            attributes = parse_filename_identifier(element.name)
            index = attributes.pop('index')
            slices.append(
                CachingSlice(filepath=element, fingerprint=fingerprint,
                             index=index)
            )
        # sort by slice index
        slices = sorted(slices, key=lambda s: s.index)
        slices = np.stack(tuple(s.data for s in slices), axis=0)
        return Subvolume(directorypath=directory, fingerprint=fingerprint,
                         data=slices)


    def from_directory(self, directory: PathLike) -> Subvolume:
        directory = Path(directory)
        attributes = postprocess(parse_directory_identifier(directory.parent.stem))
        # directory stem is expected to have the form `subvol_{N}`
        _, index = directory.stem.split('_')
        attributes['index'] = int(index)
        return self._from_directory(directory, attributes)


    def from_top_directory(self, directory: PathLike, index: int) -> Subvolume:
        directory = Path(directory)
        attributes = postprocess(parse_directory_identifier(directory.stem))
        attributes['index'] = index
        # Go one level deeper
        directory = directory / f'{self.subvolume_prefix}_{index}'
        return self._from_directory(directory, attributes)




