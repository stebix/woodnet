import pathlib
import pytest

from woodnet.checkpoint import ScoredCheckpoint

@pytest.fixture
def testfilepath(tmp_path) -> pathlib.Path:
    fpath = tmp_path / 'model-testfile.pth'
    with fpath.open(mode='w') as handle:
        handle.write('Hello World from Pytest!')
    return fpath


def test_initialization(testfilepath):
    score = 0.9 # ssio
    checkpoint = ScoredCheckpoint(score=score, filepath=testfilepath)
    print(checkpoint)


def test_failure_on_missing_file():
    nonexisting_filepath = pathlib.Path('./this-should-not-exist')
    assert not nonexisting_filepath.exists(), 'faulty test setup: nonexisting file path required'
    score = 0.9
    with pytest.raises(FileNotFoundError):
        checkpoint = ScoredCheckpoint(score=score, filepath=nonexisting_filepath)
