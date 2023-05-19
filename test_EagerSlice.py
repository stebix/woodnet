import numpy as np
import pytest

from PIL import Image

from dataobjects import CachingSlice

class MockFingerprint:
    ID = 1
    class_ = 'ahorn'
    voltage = '5kV'
    current = '100muA'
    duration = '1s'
    averages = 2


@pytest.fixture(scope='function')
def tif_file(tmp_path):
    shape = (256, 256)
    data = np.random.default_rng().integers(0, 255, size=shape)
    data = data.astype(np.int32)
    image = Image.fromarray(data)
    filepath = tmp_path / 'testslice.tif'
    image.save(filepath)
    return filepath


def test_functional_init_and_data_loading(tif_file):
    fingerprint = MockFingerprint()
    slc = CachingSlice(filepath=tif_file, fingerprint=fingerprint, index=0)
    expected_data = np.array(
        Image.open(tif_file)
    )
    assert np.allclose(slc.data, expected_data)

