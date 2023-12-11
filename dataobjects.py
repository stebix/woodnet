import dataclasses
import numpy as np
from typing import Optional 

from PIL import Image

from custom.types import PathLike


@dataclasses.dataclass(frozen=True)
class InstanceFingerprint:
    """Fingerprint of a dataset instance."""
    ID: int
    class_: str
    voltage: str
    current: str
    duration: str
    averages: int


@dataclasses.dataclass(frozen=True)
class SubvolumeFingerprint(InstanceFingerprint):
    index: int



@dataclasses.dataclass
class AbstractSlice:
    """Full info bundle on slice."""
    filepath: PathLike
    fingerprint: InstanceFingerprint
    index: int
    
    @property
    def ID(self) -> int:
        return self.fingerprint.ID
    
    @property
    def class_(self) -> str:
        return self.fingerprint.class_
    
    @property
    def data() -> np.ndarray:
        raise NotImplementedError
    
    def _load_data(self) -> np.ndarray:
        return np.array(Image.open(self.filepath))

    
@dataclasses.dataclass
class LazySlice(AbstractSlice):
    """Lazy slice loads data on access."""

    @property
    def data(self) -> np.ndarray:
        return self._load_data()
    
    
@dataclasses.dataclass
class CachingSlice(AbstractSlice):
    """Slice data loaded and cached upon request."""
    _data: Optional[np.ndarray] = dataclasses.field(init=False,
                                                    default=None,
                                                    repr=False)
    
    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = self._load_data()
        return self._data
    



def terse_data_str(cls):
    """
    Class decorator to use numpy.ndarrays shapes instead of full arrays inside
    the object string representation method.
    """
    
    def __str__(self) -> str:
        base = ''.join((type(self).__name__, '('))
        s = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                repr_value = f'(dtype={value.dtype}, shape={value.shape})'
            else:
                repr_value = repr(value)
            s.append(f'{field.name}={repr_value}')
        s = ', '.join(s)
        return ''.join((base, s, ')'))
    
    setattr(cls, '__str__', __str__)
    setattr(cls, '__repr__', __str__)
    return cls




@dataclasses.dataclass
@terse_data_str
class Volume:
    directorypath: PathLike
    fingerprint: InstanceFingerprint
    data: np.ndarray


@dataclasses.dataclass
@terse_data_str
class Subvolume:
    directorypath: PathLike
    fingerprint: SubvolumeFingerprint
    data: np.ndarray
    