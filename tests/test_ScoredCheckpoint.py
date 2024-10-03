import pathlib
import pytest
import logging

from woodnet.checkpoint import ScoredCheckpoint, excise_optimal_qualifier
from woodnet.checkpoint.scores import ScoreRank
from woodnet.checkpoint.handlers import SEP

@pytest.fixture
def testfilepath(tmp_path) -> pathlib.Path:
    fpath = tmp_path / f'model{SEP}optimal{SEP}testfile.pth'
    with fpath.open(mode='w') as handle:
        handle.write('Hello World from Pytest!')
    return fpath


def test_initialization(testfilepath):
    score = 0.9 # ssio
    rank = ScoreRank.FEASIBLE
    checkpoint = ScoredCheckpoint(score=score, filepath=testfilepath, rank=rank)
    print(checkpoint)


def test_remove_checkpoint(testfilepath):
    assert testfilepath.is_file(), 'file must exist at beginning: test setup failure'
    score = 0.9 # ssio
    rank = ScoreRank.FEASIBLE
    checkpoint = ScoredCheckpoint(score=score, filepath=testfilepath, rank=rank)
    checkpoint.remove()
    assert not testfilepath.exists(), 'should not exist after remove method call'


def test_demote_checkpoint(testfilepath):
    assert testfilepath.is_file(), 'file must exist at beginning: test setup failure'
    score = 0.9 # ssio
    rank = ScoreRank.OPTIMAL
    checkpoint = ScoredCheckpoint(score=score, filepath=testfilepath, rank=rank)
    checkpoint.demote()

    assert checkpoint.rank == ScoreRank.FEASIBLE
    assert 'optimal' not in checkpoint.filepath.name
    assert not testfilepath.exists(), 'should not exist after demotion method call'


def test_demote_checkpoint_with_nonoptimal_rank(testfilepath, caplog):
    assert testfilepath.is_file(), 'file must exist at beginning: test setup failure'
    score = 0.9 # ssio
    rank = ScoreRank.FEASIBLE
    checkpoint = ScoredCheckpoint(score=score, filepath=testfilepath, rank=rank)
    checkpoint.demote()

    # a little bit of a hacky and brittle way to check that we emitted a warning
    # about the attempted demotion of a non-optimal instance
    record = caplog.records[-1]
    assert 'attempted demotion of non-optimal instance' in record.message
    assert checkpoint.rank == ScoreRank.FEASIBLE
    assert testfilepath.exists(), 'should still exist after demotion method call'


def test_demote_checkpoint_with_nonexciseable_qualifier(tmp_path, caplog):
    fpath = tmp_path / f'picard-is%top_captain.pth'

    with fpath.open(mode='w') as handle:
        handle.write('Hello World from Pytest!')

    score = 0.9 # ssio
    rank = ScoreRank.OPTIMAL
    checkpoint = ScoredCheckpoint(score=score, filepath=fpath, rank=rank)
    checkpoint.demote()

    # a little bit of a hacky and brittle way to check that we emitted a warning
    # about the attempted demotion of a non-optimal instance
    record = caplog.records[-1]
    assert 'Qualifier excision from filename failed' in record.message
    assert checkpoint.rank == ScoreRank.OPTIMAL, 'since demotion failed, rank should not have changed'
    assert fpath.exists(), 'should still exist after demotion method call'



class Test_excise_optimal_parameter:
    # we decided upon a fixed seperator string for the paths
    # this must be observed for buidling the test and expected paths  

    def test_with_well_formed_path(self):
        testpath = pathlib.Path(
            f'/nested/directories/fold-1/chkpt{SEP}optimal{SEP}wildUUID.pth'
        )
        expected_result = pathlib.Path(
            f'/nested/directories/fold-1/chkpt{SEP}wildUUID.pth'
        )
        result = excise_optimal_qualifier(testpath)
        assert result == expected_result


    def test_with_separated_but_unusual_qualifier(self):
        testpath = pathlib.Path(
            f'/nested/directories/fold-1/chkpt{SEP}pessimal{SEP}wildUUID.pth'
        )
        expected_result = pathlib.Path(
            f'/nested/directories/fold-1/chkpt{SEP}wildUUID.pth'
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
