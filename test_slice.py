import numpy as np
import PIL
import pytest

from pathlib import Path

from dataobjects import CachingSlice, InstanceFingerprint


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
    fingerprint = InstanceFingerprint(
        ID=1701, class_='ahorn', voltage='40kV',
        current='200muA', duration='1s', averages=2
    )
    return fingerprint



def test_correct_initialization_and_data_loading_from_disk(fingerprint, tiff_generator):
    tif_path = tiff_generator.make_one()
    slc = CachingSlice(filepath=tif_path, fingerprint=fingerprint, index=0)
    expected_data = np.array(PIL.Image.open(tif_path))
    assert np.allclose(slc.data, expected_data)

