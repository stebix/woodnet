import numpy as np
import PIL
from dataobjects import CachingSlice


def test_correct_initialization_and_data_loading_from_disk(fingerprint, tiff_generator):
    tif_path = tiff_generator.make_one()
    slc = CachingSlice(filepath=tif_path, fingerprint=fingerprint, index=0)
    expected_data = np.array(PIL.Image.open(tif_path))
    assert np.allclose(slc.data, expected_data)

