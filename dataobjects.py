import dataclasses
import numpy as np
from typing import Optional 

from PIL import Image

from customtypes import PathLike


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
    

@dataclasses.dataclass
class Subvolume:
    directorypath: PathLike
    fingerprint: SubvolumeFingerprint
    data: np.ndarray
    
    @classmethod
    def load_from_directory(cls, directory: PathLike) -> 'Subvolume':
        """Instantiante `Subvolume` instance directly from directory."""
        pass