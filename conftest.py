import numpy as np
import PIL

from pathlib import Path

import pytest

from dataobjects import InstanceFingerprint

class TiffGenerator:
    """Produce physical TIFFs for testing purposes."""
    def __init__(self,
                 tmpdir_factory,
                 size: tuple[int] = (128, 128)) -> None:
        self.tmpdir_factory = tmpdir_factory
        self.tmpdir = Path(self.tmpdir_factory.mktemp('test-tiffs'))
        self.size = size
        self._counter = 0
    
    def make_many(self, n: int) -> list[Path]:
        paths = [self.make_one() for _ in range(n)]
        return paths
    
    def make_one(self) -> Path:
        data = np.random.default_rng().integers(-128, 128,
                                                size=self.size)
        data = data.astype(np.int32)
        image = PIL.Image.fromarray(data)
        path = self.tmpdir / f'randimage-{self._counter}.tif'
        self._counter += 1
        image.save(path)
        return path


@pytest.fixture(scope='session')
def tiff_generator(tmpdir_factory):
    generator = TiffGenerator(tmpdir_factory=tmpdir_factory)
    return generator


@pytest.fixture
def fingerprint() -> InstanceFingerprint:
    """Give a test fingerprint."""
    fingerprint = InstanceFingerprint(
        ID=1701, class_='ahorn', voltage='40kV',
        current='200muA', duration='1s', averages=2
    )
    return fingerprint
