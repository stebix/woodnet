import pytest

from pathlib import Path

from woodnet.configtools import load_yaml
from woodnet.configtools.validation import Trainer, TrainingConfiguration

def test_trainer_model():
    data = {
        'name' : 'TestTrainer',
        'validation_metric' : 'ACC',
        'log_after_iters' : 100,
        'max_num_epochs' : 4,
        'max_num_iters' : 100
    }
    t = Trainer(**data)


def test_fully_validated_test_configuration():
    fpath = Path(__file__).parents[1] / 'woodnet/trainconf.yaml'
    data = load_yaml(fpath)
    
    conf = TrainingConfiguration(**data)

    print('isdir :: ', conf.experiment_directory.is_dir())

