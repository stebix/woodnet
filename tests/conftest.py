"""
Provide general fixtures for the test suite.

@jsteb 2024
"""
import numpy as np
import PIL
import pytest
import zarr

from pathlib import Path

from woodnet.datasets.setup import InstanceFingerprint


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



@pytest.fixture(scope='module')
def base_data_shape() -> tuple[int, int, int, int]:
    BASE_DATA_SHAPE: tuple[int, int, int, int] = (1, 128, 128, 128)
    return BASE_DATA_SHAPE


@pytest.fixture(scope='module')
def internal_path() -> str:
    INTERNAL_PATH: str = 'group/data'
    return INTERNAL_PATH


@pytest.fixture(scope='module')
def tempdir(tmp_path_factory):
    return tmp_path_factory.mktemp('mock-raw-data')


def create_zarr_dataset(directory: Path,
                        classname: str,
                        data_shape: tuple[int, int, int, int],
                        internal_path: str,
                        dtype: np.dtype = np.float32) -> Path:
    """
    Create an authentic zarr dataset with a fingerprint inside the
    given directory. The dataset will be created with a random
    data array of shape BASE_DATA_SHAPE.

    Parameters
    ----------

    directory : Path
        The directory in which the zarr dataset should be created.
        For testing purposes, this should be a temporary directory.

    classname : str
        The class name for the dataset.

    data_shape : tuple[int, int, int, int]
        The shape of the data array that should be created.
        The format is (C, Z, Y, X) where C is the number of channels.

    internal_path : str
        The internal path of the dataset.
        For testing purposes, this is expected to be a two-part, slash-separated
        string like 'group/data'.

    Returns
    -------

    zarr_path : Path
        The path to the zarr dataset
    """
    data = np.random.default_rng().normal(0, 1, size=data_shape).astype(dtype)
    fingerprint = {
        'class_': classname,
        'voltage': '40kV',
        'current': '200muA',
        'duration': '1s',
        'averages': 2
    }
    zarr_path = directory / f'{classname}-testdataset.zarr'
    group, dataset = internal_path.split('/')
    # create internal structure and write data
    array = zarr.open(zarr_path, mode='w')
    group = array.create_group(group)
    dataset = group.create_dataset(dataset, data=data)
    # write metadata fingerprint
    for k, v in fingerprint.items():
        array.attrs[k] = v
    
    return zarr_path



@pytest.fixture(scope='module')
def mock_instance_mapping(tempdir, base_data_shape, internal_path):
    """
    Create a test-wise mock instance mapping that relates unique string IDs of
    dataset instances to their respective metadata like location and class name.

    We can modify the basal data shape and internal path for testing purposes.
    """
    class_a = 'acer'
    class_b = 'pinus'
    instance_mapping_raw = {
        'picard' : {
            'location' : str(
                create_zarr_dataset(tempdir, class_a, base_data_shape, internal_path)
                ),
            'classname' : class_a,
            'group' : 'axial'
        },
        'kirk' : {
            'location' : str(
                create_zarr_dataset(tempdir, class_b, base_data_shape, internal_path)
                ),
            'classname' : class_b,
            'group' : 'tangential'
        }
    }
    instance_mapping = {k : InstanceFingerprint(**v) for k, v in instance_mapping_raw.items()}
    return instance_mapping