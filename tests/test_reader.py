from pathlib import Path
import numpy as np
import pytest
import h5py
import zarr

from woodnet.datasets.reader import deduce_reader_class, HDF5Reader, ZarrReader

DATA_SHAPE: tuple[int, int, int, int] = (1, 25, 25, 25)
HDF5_INTERNAL_PATH: str = 'group/dataset'
ZARR_INTERNAL_PATH: str = 'group/dataset'

def create_test_data(shape: tuple[int, int, int, int]) -> np.ndarray:
    """
    Create test data with shape (C x D x H x W).
    """
    return np.random.default_rng().normal(shape)


def create_test_fingerprint() -> dict:
    return {
        'ID': 0,
        'class_': 'test',
        'instance': 'test',
        'source': 'test',
        'sourceID': 'test',
    }


@pytest.fixture
def HDF5_testdata(tmp_path) -> tuple[Path, np.ndarray, dict]:
    """Provide a realistical test HDF5 file for canonically structured data."""
    data = create_test_data(DATA_SHAPE)
    fingerprint = create_test_fingerprint()
    fpath = tmp_path / 'test.hdf5'

    with h5py.File(fpath, mode='w') as handle:
        dataset = handle.create_dataset(HDF5_INTERNAL_PATH,
                                        data=data)
        for key, value in fingerprint.items():
            dataset.attrs[key] = value
    return (fpath, data, fingerprint)


@pytest.fixture
def Zarr_testdata(tmp_path) -> tuple[Path, np.ndarray, dict]:
    """Provide a realistical test Zarr file for canonically structured data."""
    data = create_test_data(DATA_SHAPE)
    fingerprint = create_test_fingerprint()

    fpath = tmp_path / 'test.zarr'
    # our zarr layout has the special layout that the fingerprint is stored
    # in the overarching array but the data is stored in a subarray under the
    # internal path
    array = zarr.convenience.open(fpath, mode='w')
    groupname, datasetname = ZARR_INTERNAL_PATH.split('/')
    group = array.create_group(groupname)
    dataset = group.create_dataset(datasetname, data=data)
    array.attrs.update(fingerprint)
    return (fpath, data, fingerprint)


class Test_HDF5Reader:
    @pytest.mark.integration
    def test_HDF5_reader(self, HDF5_testdata):
        fpath, expected_data, expected_fingerprint = HDF5_testdata

        reader_class = deduce_reader_class(fpath)
        reader = reader_class(fpath, HDF5_INTERNAL_PATH)

        assert isinstance(reader, HDF5Reader)

        fingerprint = reader.load_fingerprint()
        data = reader.load_data()

        assert fingerprint == expected_fingerprint
        assert np.allclose(data, expected_data)



class Test_ZarrReader:
    @pytest.mark.integration
    def test_Zarr_reader(self, Zarr_testdata):
        fpath, expected_data, expected_fingerprint = Zarr_testdata

        reader_class = deduce_reader_class(fpath)
        reader = reader_class(fpath, ZARR_INTERNAL_PATH)

        assert isinstance(reader, ZarrReader)

        fingerprint = reader.load_fingerprint()
        data = reader.load_data()

        assert fingerprint == expected_fingerprint
        assert np.allclose(data, expected_data)