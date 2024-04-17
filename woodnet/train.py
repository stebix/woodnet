import os
import torch
import logging
import warnings
import tqdm.auto as tqdm

from copy import deepcopy
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import woodnet.logtools.dict.ops as logged 

from woodnet.models import create_model
from woodnet.custom.exceptions import ConfigurationError
from woodnet.datasets.volumetric import TileDatasetBuilder
from woodnet.trainer import retrieve_trainer_class
from woodnet.directoryhandlers import ExperimentDirectoryHandler
from woodnet.configtools import load_yaml, backup_configuration
from woodnet.hooks import install_loginterceptor_excepthook
from woodnet.logtools.infrastructure import (create_logging_infrastructure,
                                             finalize_logging_infrastructure,
                                             create_logfile_name)
from woodnet.extent import compute_training_extent
from woodnet.globconf import (configure_torch_cpu_threading,
                              TORCH_NUM_THREADS, TORCH_NUM_INTEROP_THREADS)


DataLoader = torch.utils.data.DataLoader

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)

BACKUP_CONFIGURATION_FILE: bool = bool(os.environ.get('BACKUP_CONFIGURATION_FILE', default=1))


def create_optimizer(model: Callable | torch.nn.Module,
                     configuration: dict) -> torch.optim.Optimizer:
    """
    Instantiate the optimizer from the top-level configuration using
    the model parameters.
    """
    if 'optimizer' not in configuration:
        raise ConfigurationError('missing required optimizer subconfiguration')

    optconf = deepcopy(configuration['optimizer'])
    name = optconf.pop('name')
    optimizer_class = getattr(torch.optim, name)
    lr = optconf.pop('learning_rate')

    optimizer = optimizer_class(model.parameters(), lr=lr, **optconf)
    return optimizer


def create_loss(configuration: dict) -> torch.nn.Module:
    """Create the loss criterion from the top-level configuration."""
    if 'loss' not in configuration:
        raise ConfigurationError('missing required loss subconfiguration')
    lossconf = deepcopy(configuration['loss'])
    name = lossconf.pop('name')
    loss_class = getattr(torch.nn, name)
    return loss_class(**lossconf)


def create_loaders(configuration: dict) -> dict[str, torch.utils.data.DataLoader]:
    """
    Create training and validation data loader from top-level config.
    """
    if 'loaders' not in configuration:
        raise ConfigurationError('missing required loaders subconfiguration')

    dataset_config = deepcopy(configuration['loaders'])

    name = dataset_config.get('dataset')
    num_workers = logged.pop(dataset_config, key='num_workers', default=1)
    pin_memory = logged.pop(dataset_config, key='pin_memory', default=False)
    batchsize = logged.pop(dataset_config, key='batchsize', default=1)
    tileshape = logged.get(dataset_config, key='tileshape', default=(128, 128, 128))

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


from woodnet.trainer import TrainerClass, retrieve_trainer_class


def get_trainer_class(configuration: dict) -> TrainerClass:
    """Extract the trainer class from the top-level configuration."""
    if 'trainer' not in configuration:
        raise ConfigurationError('missing required trainer subconfiguration')
    
    subconfig = configuration.get('trainer')
    class_name = logged.get(subconfig, 'name', 'Trainer')
    trainer_class = retrieve_trainer_class(class_name)
    return trainer_class





def create_trainer(configuration: dict,
                   model: torch.nn.Module | Callable,
                   handler: ExperimentDirectoryHandler,
                   optimizer: torch.optim.Optimizer,
                   criterion: torch.nn.Module,
                   loaders: dict[str, DataLoader],
                   validation_criterion: Callable | None = None,
                   leave_total_progress: bool = True
                   ):
    
    if 'trainer' not in configuration:
        raise ConfigurationError('missing required trainer subconfiguration')

    trainerconf = deepcopy(configuration['trainer'])
    device = logged.get(configuration, key='device', default='cpu')
    if device == 'cpu':
        logger.warning('Selected device \'cpu\' may not provided adequate runtime performance')
    
    logger.info(f'Using training device: \'{device}\'')
    
    # TODO: implement systematic retrieval for multiple trainer classes
    trainer_class_name = logged.pop(trainerconf, key='name', default='Trainer', prefix='trainer_class')
    trainer_class = retrieve_trainer_class(trainer_class_name)
    logger.info(f'Using trainer class: {trainer_class}')
    trainer_class.leave_total_progress = leave_total_progress
    logger.info(f'Using setting leave_total_progress: {leave_total_progress}')

    trainloader = loaders.get('train')
    batchsize = trainloader.batch_size
    # extract configuration values for extent specification
    conf_max_num_epochs = logged.pop(trainerconf, key='max_num_epochs', default=None)
    conf_max_num_iters = logged.pop(trainerconf, key='max_num_iters', default=None)
    conf_gradient_budget = logged.pop(trainerconf, key='gradient_budget', default=None)
    extent = compute_training_extent(loader_length=len(trainloader), max_num_epochs=conf_max_num_epochs,
                                     max_num_iters=conf_max_num_iters, gradient_budget=conf_gradient_budget,
                                     batchsize=batchsize)
    logger.info(extent)

    log_after_iters = logged.pop(trainerconf, 'log_after_iters', default=250)
    validate_after_iters = logged.pop(trainerconf, 'validate_after_iters', default=1000)
    use_amp = logged.pop(trainerconf, 'use_amp', default=True)
    use_inference_mode = logged.pop(trainerconf, 'use_inference_mode', default=True)
    save_model_checkpoint_every_n = logged.pop(trainerconf, 'save_model_checkpoint_every_n', default=5)


    if validation_criterion is None:
        validation_criterion = criterion
        logger.info(f'No validation criterion set, using deduced criterion {validation_criterion}')
    else:
        logger.info(f'Explicit validation criterion {validation_criterion} provided')
    
    validation_metric = logged.pop(trainerconf, key='validation_metric', default='ACC')
    validation_metric_higher_is_better = logged.pop(trainerconf,
                                                    key='validation_metric_higher_is_better',
                                                    default=True)
    writer = init_writer(handler=handler)    
    logger.debug(f'Injecting remaining kwargs into trainer constructor: {trainerconf}')
    
    trainer = trainer_class(
        model=model, optimizer=optimizer, criterion=criterion,
        loaders=loaders, handler=handler, validation_criterion=validation_criterion,
        device=device, max_num_epochs=extent.max_num_epochs, max_num_iters=extent.max_num_iters,
        log_after_iters=log_after_iters, validate_after_iters=validate_after_iters,
        use_amp=use_amp, use_inference_mode=use_inference_mode,
        save_model_checkpoint_every_n=save_model_checkpoint_every_n,
        validation_metric=validation_metric,
        validation_metric_higher_is_better=validation_metric_higher_is_better,
        **trainerconf
    )

    return trainer


def run_training_experiment(configuration: dict | Path | str,
                            global_torchconf: Mapping | None = {}) -> None:
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

    # configure CPU thread parallelism
    global_torchconf = global_torchconf or {}
    num_threads = global_torchconf.get('torch_num_threads', None) or TORCH_NUM_THREADS
    num_interop_threads = global_torchconf.get('torch_num_interop_threads' or TORCH_NUM_INTEROP_THREADS)
    configure_torch_cpu_threading(num_threads, num_interop_threads)

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

    if BACKUP_CONFIGURATION_FILE:
        backup_path = backup_configuration(configuration, handler, force_write=False)
        logger.debug(f'Sucessfully stored backup configuration at \'{backup_path}\'')


    logfile_path = handler.logdir / create_logfile_name()
    finalize_logging_infrastructure(logger, memoryhandler, logfile_path=logfile_path)

    model = create_model(configuration)
    logger.info(f'Successfully created model with string dump: {model}')

    optimizer = create_optimizer(model, configuration)
    logger.info(f'Created optimizer: {optimizer}')

    criterion = create_loss(configuration)
    logger.info(f'Created criterion: {criterion}')

    loaders = create_loaders(configuration)
    device = torch.device(logged.get(configuration, key='device', default='cpu'))

    trainer_class = get_trainer_class(configuration)
    trainer = trainer_class.create(
        configuration=configuration,
        model=model,
        handler=handler,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        loaders=loaders,
        validation_criterion=None,
        leave_total_progress=True
    )
    logger.info(f'Successfully created trainer object. initializing core training loop')
    trainer.train()

    logger.info('Successfully concluded train method.')


def crawl_configurations(directory: Path, /) -> list[dict]:
    results = []
    for item in directory.iterdir():
        if item.suffix in {'.yaml', '.yml'}:
            logger.info(f'Loaded element \'{item}\' during configuration crawling')
            results.append(load_yaml(item))
        else:
            logger.debug(f'Ignored element \'{item}\' during configuration crawling')
    return results


def precheck_configuration_batch(configurations: Iterable[dict]) -> None:
    """Perform existence of experiment directory and device homogeneity precheck."""
    devices = []
    for configuration in configurations:
        if 'experiment_directory' not in configuration:
            raise ConfigurationError('missing required experiment directory specification')
        devices.append(configuration['device'])
    devices = set(devices)
    if len(devices) > 1:
        warnings.warn(f'Experiment batch runs on multiple devices: {devices}! Make '
                      f'sure to properly coordinate resources to avoid device contention.')


def actualize_configurations(configurations: Iterable[dict | Path | str]) -> list[dict]:
    configurations = list(configurations)
    actualized = []
    for i, configuration in enumerate(configurations):
        if isinstance(configuration, (str, Path)):
            path = Path(configuration)

            if path.is_file():
                logger.debug(f'Loading configuration {i}/{len(configurations)} from file '
                             f'system location \'{configuration}\'')
                actualized.append(load_yaml(configuration))

            elif path.is_dir():
                logger.debug(f'Discovering configurations in directory \'{path}\'')
                actualized.extend(crawl_configurations(path))
            
            else:
                logger.debug(f'Ignored nonexisting configuration location \'{path}\'')

        else:
            actualized.append(configuration)
    return actualized



def run_training_experiment_batch(configurations: Iterable[dict | Path | str],
                                  global_torchconf: Mapping | None = None) -> None:

    levels = {
        'DEBUG' : logging.DEBUG,
        'INFO' : logging.INFO,
        'WARNING' : logging.WARNING,
        'ERROR' : logging.ERROR,
        'CRITICAL' : logging.CRITICAL
    }

    stream_log_level = levels.get(os.environ.get('STREAM_LOG_LEVEL', None), logging.ERROR)

    level = logging.DEBUG
    logger, streamhandler, memoryhandler = create_logging_infrastructure(level=level,
                                                                         streamhandler_level=stream_log_level)

    install_loginterceptor_excepthook(logger)

    # configure CPU thread parallelism
    global_torchconf = global_torchconf or {}
    num_threads = global_torchconf.get('torch_num_threads', None) or TORCH_NUM_THREADS
    num_interop_threads = global_torchconf.get('torch_num_interop_threads' or TORCH_NUM_INTEROP_THREADS)
    configure_torch_cpu_threading(num_threads, num_interop_threads)

    configurations = actualize_configurations(configurations)
    precheck_configuration_batch(configurations)

    # TODO: Remove this in production
    ExperimentDirectoryHandler.allow_preexisting_dir = True
    ExperimentDirectoryHandler.allow_overwrite = True
    kwargs = {'desc' : 'configuration batch progress', 'leave' : True, 'unit' : 'conf'}
    for configuration in tqdm.tqdm(configurations, **kwargs):

        handler = ExperimentDirectoryHandler(configuration['experiment_directory'])

        if BACKUP_CONFIGURATION_FILE:
            backup_path = backup_configuration(configuration, handler, force_write=False)
            logger.debug(f'Sucessfully stored backup configuration at \'{backup_path}\'')

        logfile_path = handler.logdir / create_logfile_name()
        filehandler = finalize_logging_infrastructure(logger, memoryhandler, logfile_path=logfile_path)

        model = create_model(configuration)
        logger.info(f'Successfully created model with string dump: {model}')

        optimizer = create_optimizer(model, configuration)
        logger.info(f'Created optimizer: {optimizer}')

        criterion = create_loss(configuration)
        logger.info(f'Created criterion: {criterion}')

        loaders = create_loaders(configuration)

        device = torch.device(logged.get(configuration, key='device', default='cpu'))

        trainer_class = get_trainer_class(configuration)
        trainer = trainer_class.create(
            configuration=configuration,
            model=model,
            handler=handler,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            loaders=loaders,
            validation_criterion=None,
            leave_total_progress=False
        )

        logger.info(f'Succesfully created trainer object. initializing core training loop')
        trainer.train()
        logger.info('Succesfully concluded train method.')

        # logging handler cleanup to avoid double logging
        logger.removeHandler(filehandler)
        logger.addHandler(memoryhandler)




