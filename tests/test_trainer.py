import os
import logging

from collections.abc import Mapping
from pathlib import Path

import torch
import pytest
from torch.utils.data import DataLoader

from woodnet.directoryhandlers import ExperimentDirectoryHandler
from woodnet.trainer.base_trainer import Trainer
from woodnet.trainer import retrieve_trainer_class

# allow user running the test from CLI to set the log level programmatically
# as an environment variable
ENV_LOG_LEVEL: str = os.environ.get('LOGLEVEL', default='INFO')
LEVEL_MAPPING: dict  = {
    'DEBUG' : logging.DEBUG,
    'INFO' : logging.INFO,
    'WARNING' : logging.WARNING,
    'ERROR' : logging.ERROR,
    'CRITICAL' : logging.CRITICAL
}


@pytest.fixture
def experiment_directory(tmp_path) -> Path:
    """
    Create path for experiment directory.
    """
    exp_dir = tmp_path / 'test_experiment' / 'fold-1' 
    return exp_dir



class TinyFixtureModel(torch.nn.Module):
    """Mini 3D CNN"""
    def __init__(self) -> None:
        super().__init__()
        self.dimensionality = '3D'
        self.testing = False
        # layer definitions
        self.layer_1 = torch.nn.Conv3d(
            kernel_size=3,
            in_channels=1, out_channels=64
        )
        self.relu = torch.nn.ReLU()
        self.layer_2 = torch.nn.Conv3d(
            kernel_size=3,
            in_channels=64, out_channels=32
        )
        self.avgpool = torch.nn.AdaptiveAvgPool3d(
            output_size=(2, 2, 2)
        )
        self.linear = torch.nn.Linear(
            in_features=32*2**3, out_features=1
        )
        self.final_nonlinearity = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        if self.testing:
            x = self.final_nonlinearity(x)
        
        return x


class FixtureDataset(torch.utils.data.Dataset):
    """Emulates 3D volume dataset for binary classification."""
    def __init__(self,
                 chunksize: tuple[int, int, int] = (32, 32, 32),
                 length: int = 100,
                 dtype: torch.dtype = torch.float32):

        self.chunksize = chunksize
        self.length = length
        self.dtype = dtype

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index >= self.length:
            raise IndexError(f'index {index} out of range for dataset '
                             f'with length {len(self)}')
        
        data = torch.randn(size=self.chunksize, dtype=self.dtype).unsqueeze_(0)
        label = torch.randint(0, 2, size=(1,)).to(self.dtype)
        return (data, label)
        
    def __len__(self) -> int:
        return self.length

@pytest.fixture
def loaders() -> dict[str, DataLoader]:
    train_loader = DataLoader(FixtureDataset(), batch_size=3, shuffle=True)
    val_loader = DataLoader(FixtureDataset(), batch_size=3, shuffle=False)
    return {'train' : train_loader, 'val' : val_loader}


@pytest.fixture
def configuration() -> dict:
    registry_conf = {'name' : 'Registry', 'capacity' : 4,
                     'score_preference' : 'higher_is_better'}
    
    paramlogger_conf = {'name' : 'HistogramLogger'}

    conf = {
        'name' : 'Trainer',
        'max_num_epochs' : 100,
        'max_num_iters' : 1000,
        'log_after_iters' : 25,
        'validate_after_iters' : 50,
        'use_amp' : True,
        'use_inference_mode' : True,
        'save_model_checkpoint_every_n' : 100,
        'validation_metric' : 'ACC',
        'score_registry' : registry_conf,
        'parameter_logger' : paramlogger_conf
    }
    return {'trainer' : conf}


class Test_retrieve_trainer_class:

    def test_with_Trainer(self):
        name = 'Trainer'
        expected_class = Trainer
        class_ = retrieve_trainer_class(name)
        assert class_ == expected_class



@pytest.mark.integration
def test_trainer_initialization(configuration,
                                loaders,
                                experiment_directory,
                                caplog
                                ):
    
    caplog.set_level(LEVEL_MAPPING[ENV_LOG_LEVEL])

    # settings
    lr = 1e-3
    device = torch.device('cuda')
    
    criterion = torch.nn.BCEWithLogitsLoss()
    validation_criterion = criterion
    handler = ExperimentDirectoryHandler(experiment_directory)
    model = TinyFixtureModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    trainer = Trainer.create(configuration=configuration,
                             model=model,
                             handler=handler,
                             device=device,
                             optimizer=optimizer,
                             criterion=criterion,
                             loaders=loaders,                             
                             validation_criterion=validation_criterion
                             )
    
    print(trainer)
    trainer.train()

    