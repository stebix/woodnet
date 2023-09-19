import torch
import logging

from copy import deepcopy
from collections.abc import Callable
from typing import Any, Hashable, Literal
from pathlib import Path

from woodnet.models import create_model
from woodnet.custom.exceptions import ConfigurationError
from woodnet.datasets.volumetric import TileDataset, TileDatasetBuilder
from woodnet.training import AbstractBaseTrainer, retrieve_trainer_class
from woodnet.directoryhandlers import ExperimentDirectoryHandler
from woodnet.configtools import load_yaml
from woodnet.hooks import install_loginterceptor_excepthook

DataLoader = torch.utils.data.DataLoader

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def create_optimizer(model: Callable | torch.nn.Module,
                     configuration: dict) -> torch.optim.Optimizer:
    """
    Instantiate the optimimzer from the top-level configuration using
    the model parameters.
    """
    if 'optimizer' not in configuration:
        raise ConfigurationError('missing required optimimzer subconfiguration')

    optconf = deepcopy(configuration['optimizer'])
    name = optconf.pop('name')
    optimizer_class = getattr(torch.optim, name)
    lr = optconf.pop('learning_rate')

    optimizer = optimizer_class(model.parameters(), lr=lr, **optconf)
    return optimizer



def create_loss(configuration: dict) -> torch.nn.Module:
    """Cereate the loss criterion from the top-level configuration."""
    if 'loss' not in configuration:
        raise ConfigurationError('missing required loss subconfiguration')
    lossconf = deepcopy(configuration['loss'])
    name = lossconf.pop('name')
    loss_class = getattr(torch.nn, name)
    return loss_class(**lossconf)



def retrieve_logged(d: dict, /, key: Hashable, default: Any, method: Literal['get', 'pop'],
                    prefix: str = '', suffix: str = '') -> Any:
    """
    Retrieve a value of dictionary and log whether actual or default value was retrieved.

    Parameters
    ----------

    d : dictionary
        Base container.

    key : Hashable
        Key for the desired value.

    default : Any
        Default value returned if key - value pair is not present.

    method : Literal
        Choose between value get (non-modifying) and value pop
        (in-place modfication).

    prefix : str, optional
        Set additional prefix information for the key.
        Defaults to empty string. 

    suffix : str, optional
        Set additional suffix information for the key.
        Defaults to empty string. 

    Returns
    -------

    value : Any
        Either the value defined by the dictionary or the default value.
    """
    sentinel = object()
    retrieve = d.get if method == 'get' else d.pop
    value = retrieve(key, sentinel)
    expanded_key = ''.join((prefix, key, suffix))
    if value is sentinel:
        value = default
        logger.info(f'Using \'{expanded_key}\' with internal default value < {value} >')
    else:
        logger.info(f'Using \'{expanded_key}\' with configuration value < {value} >')
    return value


def get_logged(d: dict, /, key: Hashable, default: Any,
               prefix: str = '', suffix: str = '') -> Any:
    return retrieve_logged(d, key=key, default=default, method='get',
                           prefix=prefix, suffix=suffix)

def pop_logged(d: dict, /, key: Hashable, default: Any,
               prefix: str = '', suffix: str = '') -> Any:
    return retrieve_logged(d, key=key, default=default, method='pop',
                           prefix=prefix, suffix=suffix)



def create_loaders(configuration: dict) -> dict[str, torch.utils.data.DataLoader]:
    """
    Create training and validation data loader from top-level config.
    """
    if 'loaders' not in configuration:
        raise ConfigurationError('missing required loaders subcongiguration')

    dataset_config = deepcopy(configuration['loaders'])

    name = dataset_config.get('dataset')
    num_workers = pop_logged(dataset_config, key='num_workers', default=1)
    pin_memory = pop_logged(dataset_config, key='pin_memory', default=False)
    batchsize = pop_logged(dataset_config, key='batchsize', default=1)
    tileshape = get_logged(dataset_config, key='tileshape', default=(128, 128, 128))

    # TODO: change this to clear interface for all dataset types
    assert name == 'TileDataset'
    builder = TileDatasetBuilder()

    loaders = {}

    for phase in ['train', 'val']:
        
        phase_config = dataset_config.get(phase)
        phase_config.update({'tileshape' : tileshape, 'phase' : phase})
        datasets = builder.build(**phase_config)
        dataset = torch.utils.data.ConcatDataset(datasets)
        shuffle = True if phase == 'train' else False

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batchsize, num_workers=num_workers,
            shuffle=shuffle, pin_memory=pin_memory
        )
        loaders[phase] = loader

    return loaders
    


def create_trainer(configuration: dict,
                   model: torch.nn.Module | Callable,
                   handler: ExperimentDirectoryHandler,
                   optimizer: torch.optim.Optimizer,
                   criterion: torch.nn.Module,
                   loaders: dict[str, DataLoader],
                   validation_criterion: Callable | None = None
                   ) -> AbstractBaseTrainer:
    
    if 'trainer' not in configuration:
        raise ConfigurationError('missing required trainer subconfiguration')

    trainerconf = deepcopy(configuration['trainer'])
    device = get_logged(configuration, key='device', default='cpu')
    if device == 'cpu':
        logger.warning('Selected device \'cpu\' may not provided adequate runtime performance')
    
    logger.info(f'Using training device: \'{device}\'')
    
    # TODO: implement systematic retrieval for multiple trainer classes
    trainer_class_name = pop_logged(trainerconf, key='name', default='Trainer', prefix='trainer_class')
    trainer_class = retrieve_trainer_class(trainer_class_name)
    logger.info(f'Using trainer class: {trainer_class}')

    # retrieve other required information
    max_num_epochs = trainerconf.pop('max_num_epochs')
    max_num_iters = trainerconf.pop('max_num_iters')

    log_after_iters = pop_logged(trainerconf, 'log_after_iters', default=250)
    validate_after_iters = pop_logged(trainerconf, 'validate_after_iters', default=1000)
    use_amp = pop_logged(trainerconf, 'use_amp', default=True)
    use_inference_mode = pop_logged(trainerconf, 'use_inference_mode', default=True)
    save_model_checkpoint_every_n = pop_logged(trainerconf, 'save_model_checkpoint_every_n', default=5)


    if validation_criterion is None:
        validation_criterion = criterion
        logger.info(f'No validation criterion set, using deduced criterion {validation_criterion}')
    else:
        logger.info(f'Explicit validation criterion {validation_criterion} provided')
    
    
    trainer = trainer_class(
        model=model, optimizer=optimizer, criterion=criterion,
        loaders=loaders, handler=handler, validation_criterion=validation_criterion,
        device=device, max_num_epochs=max_num_epochs, max_num_iters=max_num_iters,
        log_after_iters=log_after_iters, validate_after_iters=validate_after_iters,
        use_amp=use_amp, use_inference_mode=use_inference_mode,
        save_model_checkpoint_every_n=save_model_checkpoint_every_n
    )

    return trainer

from woodnet.logtools import (create_logging_infrastructure, finalize_logging_infrastructure,
                              create_logfile_name)


def run_training_experiment(configuration: dict | Path | str) -> None:
    """Run a training experiment fully defined by the configuration.

    The configuration may be provided as a dictionary or mapping to be directly consumed.
    Otherwise, e.g. if a string or pathlib.path instance, it is interpreted as a path-like
    object that points to a YAML file.

    The function sets up all necessary subcomponents, collects them into the trainer
    object and subsequently runs the core training loop.
    """
    level = logging.DEBUG
    logger, streamhandler, memoryhandler = create_logging_infrastructure(level=level)

    install_loginterceptor_excepthook(logger)

    if isinstance(configuration, (str, Path)):
        configuration = Path(configuration)
        logger.debug(f'Loading configuration from file system location \'{configuration}\'')
        configuration = load_yaml(configuration)

    if 'experiment_directory' not in configuration:
        raise ConfigurationError('missing required experiment directory specification')

    # TODO: Remove this in production
    ExperimentDirectoryHandler.allow_preexisting_dir = True
    ExperimentDirectoryHandler.allow_overwrite = True

    handler = ExperimentDirectoryHandler(configuration['experiment_directory'])

    logfile_path = handler.logdir / create_logfile_name()
    finalize_logging_infrastructure(logger, memoryhandler, logfile_path=logfile_path)

    model = create_model(configuration)
    logger.info(f'Sucessfully created model with string dump: {model}')

    optimizer = create_optimizer(model, configuration)
    logger.info(f'Created optimizer: {optimizer}')

    criterion = create_loss(configuration)
    logger.info(f'Created criterion: {criterion}')

    loaders = create_loaders(configuration)

    trainer = create_trainer(configuration=configuration, model=model, optimizer=optimizer,
                             criterion=criterion, loaders=loaders, handler=handler,
                             validation_criterion=None)

    logger.info(f'Succesfully created trainer object. initializing core training loop')
    trainer.train()

    logger.info('Sucesffuly concluded train method.')




