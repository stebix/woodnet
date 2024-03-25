import pathlib
import pytest

from woodnet.checkpoint import ScoredCheckpoint, excise_optimal_qualifier
from woodnet.checkpoint.scores import ScoreRank

@pytest.fixture
def testfilepath(tmp_path) -> pathlib.Path:
    fpath = tmp_path / 'model-optimal-testfile.pth'
    with fpath.open(mode='w') as handle:
        handle.write('Hello World from Pytest!')
    return fpath


def test_initialization(testfilepath):
    score = 0.9 # ssio
    rank = ScoreRank.FEASIBLE
    checkpoint = ScoredCheckpoint(score=score, filepath=testfilepath, rank=rank)
    print(checkpoint)


def test_remove_file(testfilepath):
    assert testfilepath.is_file(), 'file must exist at beginning: test setup failure'
    score = 0.9 # ssio
    rank = ScoreRank.FEASIBLE
    checkpoint = ScoredCheckpoint(score=score, filepath=testfilepath, rank=rank)
    checkpoint.remove()
    assert not testfilepath.exists(), 'should not exist after remove method call'


def test_demote_file(testfilepath):
    assert testfilepath.is_file(), 'file must exist at beginning: test setup failure'
    score = 0.9 # ssio
    rank = ScoreRank.OPTIMAL
    checkpoint = ScoredCheckpoint(score=score, filepath=testfilepath, rank=rank)
    checkpoint.demote()

    assert checkpoint.rank == ScoreRank.FEASIBLE
    assert 'optimal' not in checkpoint.filepath.name
    assert not testfilepath.exists(), 'should not exist after demotion method call'



class Test_excise_optimal_parameter:

    def test_with_well_formed_path(self):
        testpath = pathlib.Path(
            '/nested/directories/fold-1/chkpt-optimal-wildUUID.pth'
        )
        expected_result = pathlib.Path(
            '/nested/directories/fold-1/chkpt-wildUUID.pth'
        )
        result = excise_optimal_qualifier(testpath)
        assert result == expected_result


    def test_with_separated_but_unusual_qualifier(self):
        testpath = pathlib.Path(
            '/nested/directories/fold-1/chkpt-pessimal-wildUUID.pth'
        )
        expected_result = pathlib.Path(
            '/nested/directories/fold-1/chkpt-wildUUID.pth'
        )
        result = excise_optimal_qualifier(testpath)
        assert result == expected_result


    def test_with_malformed_path(self):
        """
        Expected to fail due to missing dash-separated qualifier 
        in between prefix and UUID part.
        """
        testpath = pathlib.Path(
            '/nested/directories/fold-1/chkpt-wildUUID.pth'
        )
        with pytest.raises(ValueError):
            _ = excise_optimal_qualifier(testpath)


@pytest.mark.skip
def test_failure_on_missing_file():
    nonexisting_filepath = pathlib.Path('./this-should-not-exist')
    assert not nonexisting_filepath.exists(), 'faulty test setup: nonexisting file path required'
    score = 0.9
    rank = ScoreRank.OPTIMAL
    with pytest.raises(FileNotFoundError):
        checkpoint = ScoredCheckpoint(score=score, filepath=nonexisting_filepath,
                                      rank=rank)
