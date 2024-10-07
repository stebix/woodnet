import time
from pathlib import Path

import pytest

from woodnet.inference.directories import (TrainingResultBag, are_consecutive,
                                           get_fold_directories,
                                           CrossValidationResultsBag)


# TODO: maybe import this from the actual file to centralize the layout information
CHECKPOINT_DIR_NAME: str = 'checkpoints'
LOG_DIR_NAME: str = 'logs'

TENSORBOARD_FILE_PREFIX: str = 'events.out.tfevents'
CHECKPOINT_FILE_SUFFIX: str = 'pth'
LOG_FILE_SUFFIX: str = 'log'
TRAINING_CONFIG_SUFFIX: str = 'yaml'



def make_checkpoints(dir: Path, n: int = 3) -> Path:
    assert dir.is_dir(), 'make_checkpoints requires pre-existing directory'
    prefix = 'chkpt'
    suffix = '.pth'
    for i in range(n):
        medfix = '_optimal' if i == 0 else ''
        filename = f'{prefix}{medfix}_{i}{suffix}'
        filepath = dir / filename
        with filepath.open(mode='w') as handle:
            handle.write(f'Hello world from checkpoint mock file :: N = {i}')
    return filepath


def make_logfile(dir: Path) -> Path:
    assert dir.is_dir(), 'make_logfile requires pre-existing directory'
    fname = '.'.join((str(time.time()), LOG_FILE_SUFFIX))
    filepath = dir / fname
    with filepath.open(mode='w') as handle:
        handle.write(f'Hello world from mock log file @ {time.time()}')
    return filepath
    

def make_tensorboard_file(dir: Path) -> Path:
    assert dir.is_dir(), 'make_logfile requires pre-existing directory'
    fname = '.'.join((TENSORBOARD_FILE_PREFIX, str(time.time())))
    filepath = dir / fname
    with filepath.open(mode='w') as handle:
        handle.write(f'Hello world from tensorboard file @ {time.time()}')
    return filepath


def make_configuration_file(dir: Path) -> None:
    assert dir.is_dir(), 'make_logfile requires pre-existing directory'
    fname = '.'.join(('mock_training_configuration', TRAINING_CONFIG_SUFFIX))
    filepath = dir / fname
    with filepath.open(mode='w') as handle:
        handle.write(f'Hello world from mock training configuration file @ {time.time()}')
    return filepath


def populate_mock_training_directory(basedir: Path) -> Path:
    basedir.mkdir(exist_ok=True)
    ckpt_dir = basedir / CHECKPOINT_DIR_NAME
    log_dir = basedir / LOG_DIR_NAME
    for d in (ckpt_dir, log_dir):
        d.mkdir()
    make_checkpoints(ckpt_dir)
    make_configuration_file(log_dir)
    make_logfile(log_dir)
    make_tensorboard_file(log_dir)
    return basedir


def populate_mock_cv_directory(basedir: Path, foldcount: int = 3) -> list[Path]:
    """Mock cross validation fold directory."""
    fold_dirs = [ ]
    for i in range(1, foldcount+1):
        d = basedir / f'fold-{i}'
        d.mkdir() 
        fold_dirs.append(d)
    return fold_dirs



@pytest.fixture
def training_directory(tmp_path) -> Path:
    """Create realistic training directory with appropriately named file contents"""
    traindir = tmp_path / 'test-train-dir'
    traindir = populate_mock_training_directory(traindir)
    return traindir


class Test_TrainingResultBag:

    def test_from_directory(self, training_directory):
        result_bag = TrainingResultBag.from_directory(training_directory)
        assert len(result_bag.checkpoints) == 3




class Test_get_fold_directories:
    @pytest.mark.parametrize('foldcount', (3, 5))
    def test_with_three_valid_subdirs(self, tmp_path, foldcount):
        basedir = tmp_path
        _ = populate_mock_cv_directory(basedir, foldcount=foldcount)
        result = get_fold_directories(basedir)
        assert len(result) == foldcount
        assert result.keys() == set(range(1, foldcount+1))




class Test_are_consecutive:

    def test_with_presorted_consecutive_input(self):
        nums = list(range(1, 4))
        assert are_consecutive(nums)
    
    def test_with_unsorted_consecutive_input(self):
        nums = [3, 1, 2]
        assert are_consecutive(nums)

    def test_with_non_consecutive_input(self):
        nums = [1, 2, 4]
        assert not are_consecutive(nums)
    




class Test_CrossValidationResultsBag:

    @pytest.mark.parametrize('foldcount', (3, 5))
    def test_smoke(self, tmp_path, foldcount):
        basedir = tmp_path
        # setup mock training directory
        folddirs = populate_mock_cv_directory(basedir, foldcount=foldcount)
        for folddir in folddirs:
            populate_mock_training_directory(folddir)

        cvrb = CrossValidationResultsBag.from_directory(basedir)
        assert cvrb.foldcount == foldcount
        

    @pytest.mark.parametrize('foldcount', (3, 5))
    def test_retrieve_model_instances(self, tmp_path, foldcount):
        basedir = tmp_path
        # setup mock training directory
        folddirs = populate_mock_cv_directory(basedir, foldcount=foldcount)
        for folddir in folddirs:
            populate_mock_training_directory(folddir)

        cvrb = CrossValidationResultsBag.from_directory(basedir)
        import rich
        rich.print(cvrb.retrieve_model_instances_locations())
        
