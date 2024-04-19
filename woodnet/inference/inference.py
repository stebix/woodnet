"""
Facilitate inference runs.


@jsteb 2024
"""
import logging
import torch

import torch.amp
import torch.utils
import tqdm

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from woodnet.inference.parametrized_transforms import (CongruentParametrizations, Parametrizations,
                                                       generate_parametrized_transforms)

from woodnet.inference.directories import CrossValidationResultsBag, create_inference_directory

from woodnet.datasets.volumetric_inference import TransformedTileDatasetBuilder
from woodnet.datasets.planar import TransformedEagerSliceDatasetBuilder

from woodnet.custom.exceptions import ConfigurationError
from woodnet.logtools.dict import LoggedDict
from woodnet.configtools.validation import TrainingConfiguration

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def maybe_wrap_loader(loader: torch.utils.data.DataLoader,
                      display_progress: bool):
    if not display_progress:
        return loader
    settings = {
        'desc' : 'loader progress', 'unit' : 'bt',
        'leave' : False
    }
    return tqdm.tqdm(loader, **settings)


def get_transforms_name(transforms) -> str:
    """Try to get/deduce the global semantical name of the transforms."""
    try:
        name = transforms.name
    except AttributeError:
        name = 'parametrized transforms'
    return name


def create_parametrized_transforms(configuration: Mapping) -> list[Parametrizations] | None:
    specifications = configuration.get('parametrized_transforms', None)
    if not specifications:
        return None
    return generate_parametrized_transforms(*specifications, squeeze=False)


def create_loader(configuration: Mapping) -> torch.utils.data.DataLoader:
    """
    Create the inference data loader from the top-level configuration mapping object.
    Note: At this point no parametrized transforms are attached to the underlying datasets.

    Parameters
    ----------

    configuration : Mapping
        Top-level configuration mapping.

    
    Returns
    -------

    loader : torch.utils.data.DataLoader
        Configured and usable data loader instance.    
    """
    if 'loaders' not in configuration:
        raise ConfigurationError('missing required loaders subconfiguration')
    
    loaders_config = LoggedDict(deepcopy(configuration['loaders']), logger)
    name: str = loaders_config.pop('dataset')
    num_workers: int = loaders_config.get('num_workers', default=1)
    pin_memory: bool = loaders_config.pop('pin_memory', default=False)
    batch_size: int = loaders_config.pop('batchsize', default=1)
    tileshape: tuple[int, int, int] = loaders_config.pop('tileshape', default=(128, 128, 128))

    assert name == 'TransformedTileDataset'
    builder = TransformedTileDatasetBuilder()

    phase = 'val'
    phase_config = deepcopy(loaders_config[phase])
    phase_config.update({'tileshape' : tileshape})
    datasets = builder.build(**phase_config)
    dataset = torch.utils.data.ConcatDataset(datasets)
    shuffle = False

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=num_workers, shuffle=shuffle,
                                         pin_memory=pin_memory)

    return loader



def extract_IDs(configuration: Mapping) -> list[str]:
    """
    Extract the list of validation dataset IDs from a top-level
    (training) configuration.

    Usage hint: Can be utilized to automate the generation of the inference
    configuration from the training configuration. 
    """
    configuration = TrainingConfiguration(**configuration)
    return configuration.loaders.val.instances_ID
    


def extract_model_config(configuration: Mapping) -> tuple[dict, dict]:
    """
    Extract the model and compile configuration from the top-level training
    configuration.

    Returns
    -------

    confs : tuple[dict, dict]
        Core model configuration (0-th element) and compilation
        configuration (1-th element).  
    """
    configuration = TrainingConfiguration(**configuration)
    modelconf = configuration.model.model_dump()
    compileconf = modelconf.pop('compile', {})
    return (modelconf, compileconf)


def deduce_loader_from_training(configuration: Mapping,
                                batch_size: int,
                                num_workers: int,
                                pin_memory: bool
                                ) -> torch.utils.data.DataLoader:
    """
    Produce a usable data loader via deduction from the training configuration.
    """
    conf = TrainingConfiguration(**configuration)

    datasetname2builder: dict[str, callable] = {
        'TileDataset' : TransformedTileDatasetBuilder,
        'EagerSliceDataset' : TransformedEagerSliceDatasetBuilder,
        'TriaxialDataset' : NotImplemented
    }

    instances_ID = conf.loaders.val.instances_ID
    transform_configurations = [
        elem.model_dump() for elem in conf.loaders.val.transform_configurations
    ]
    kwargs = conf.loaders.dataset_kwargs()

    logger.info(f'Deduced instance ID set: {instances_ID}')
    logger.info(f'Deduced transform configurations: {transform_configurations}')

    builder_class = datasetname2builder[conf.loaders.dataset]
    builder = builder_class()

    datasets = builder.build(instances_ID=instances_ID, #phase='val',
                             transform_configurations=transform_configurations,
                             **kwargs)
    
    dataset = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=num_workers, pin_memory=pin_memory,
                                         shuffle=False)
    return loader



def transmogrify_state_dict(state_dict: Mapping) -> OrderedDict:
    """
    Try to turn state dict from compiled model into normal state dict.
    """
    prefix: str = '_orig_mod.'
    return OrderedDict({key.removeprefix(prefix) : value for key, value in state_dict.items()})



def run_prediction_evaluation(basedir: Path,
                              transforms: Sequence[CongruentParametrizations],
                              device: torch.device,
                              dtype: torch.dtype,
                              use_amp: bool,
                              use_inference_mode: bool,
                              display_fold_progress: bool = True,
                              leave_fold_progress: bool = True,
                              display_model_instance_progress: bool = True,
                              display_transforms_progress: bool = True,
                              display_parametrizations_progress: bool = True,
                              display_loader_progress: bool = True,                       
                              loader_worker_count: int = 0
                              ) -> dict:
    """
    Run full evaluation for a CV-fold training result.
    """
    cv_results_bag = CrossValidationResultsBag.from_directory(basedir)
    inference_directory = create_inference_directory(basedir)