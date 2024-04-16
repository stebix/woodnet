"""
Tools to adjust cross validation folds/splits in configuration objects and files.

@jsteb 2023
"""
import logging

from copy import deepcopy
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from ruamel.yaml import CommentedMap

from woodnet.configtools import load_yaml, write_yaml
from woodnet.configtools.validation import TrainingConfiguration
from woodnet.cvsplits import strategy_to_generator, CVStrategy
from woodnet.custom.types import PathLike



ConfigurationLike = Mapping | TrainingConfiguration

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


def set_instances_ID_mapping(configuration: dict,
                             training_instances: Iterable[str],
                             validation_instances: Iterable[str]) -> dict:
    """Set the instance IDs of a raw configuration dictionary."""
    configuration = deepcopy(configuration)
    configuration['loaders']['train']['instances_ID'] = list(training_instances)
    configuration['loaders']['val']['instances_ID'] = list(validation_instances)
    return configuration


def set_instances_ID_validated(configuration: TrainingConfiguration,
                               training_instances: Iterable[str],
                               validation_instances: Iterable[str]) -> dict:
    """Set the instance IDs of a validated configuration object."""
    configuration.loaders.train.instances_ID = list(training_instances)
    configuration.loaders.val.instances_ID = list(validation_instances)
    return configuration


set_instances_ID_dispatcher: dict[type, Callable] = {
    Mapping : set_instances_ID_mapping,
    TrainingConfiguration : set_instances_ID_validated,
    CommentedMap : set_instances_ID_mapping
}


def set_instances_ID(configuration: dict | TrainingConfiguration,
                     training_instances: Iterable[str],
                     validation_instances: Iterable[str]) -> TrainingConfiguration:
    """Set the instance IDs of a training configuration.

    The function accepts raw configurations (i.e. dictionaries of correct layout)
    and validated configurations (pydantic model objects).
    """
    try:
        handler = set_instances_ID_dispatcher[type(configuration)]
    except KeyError as e:
        raise KeyError(f'could not find set_instances_ID handler for '
                       f'configuration argument of {type(configuration)}') from e
    configuration = handler(configuration, training_instances, validation_instances)
    return configuration


def parse_fold_directory(path: PathLike) -> tuple[str, int]:
    """Parse and validate the fold experiment directory.
    
    Raises ValueError if non-conforming argument is received.
    """
    path = Path(path) if not isinstance(path, Path) else path
    try:
        prefix, foldnum = path.name.split('-')
    except ValueError as e:
        raise ValueError(f'Could not split tentative fold directory: \'{path.name}\'') from e
    
    if prefix != 'fold':
        raise ValueError(f'Primary directory name prefix \'{prefix}\' is not \'fold\'')
    
    try:
        foldnum = int(foldnum)
    except ValueError as e:
        raise ValueError(f'Tentative fold number \'{foldnum}\' could not be cast as integer') from e
    
    return (prefix, foldnum)



def generate_fold_directory(p: PathLike, foldnum: int) -> str:
    """
    Generate the path to the directory of the indicated n-th fold.

    The basal path is taken from the input path instance.
    """
    p = Path(p)
    try:
        prefix, previous_foldnum = parse_fold_directory(p)
    except ValueError:
        message = (f'Could not parse \'{p}\' as fold-wise experiment directory. Appending '
                   f'fold-wise experiment directory!')
        logger.warning(message)
        dirpath = p / f'fold-{foldnum}'
    else:
        if foldnum != (previous_foldnum + 1):
            message = (f'indicated fold number {foldnum} is non-consecutive to '
                       f'previous: {previous_foldnum}')
            logger.warning(message)
        dirpath = p.parent / f'{prefix}-{foldnum}'
    return str(dirpath)



def update_fold_directory(configuration: ConfigurationLike, foldnum: int) -> None:
    """
    In-place update the fold directory based with the indicated fold number.
    """
    if isinstance(configuration, Mapping):
        experiment_directory = configuration['experiment_directory']
        updated_experiment_directory = generate_fold_directory(experiment_directory, foldnum)
        configuration['experiment_directory'] = updated_experiment_directory

    elif isinstance(configuration, TrainingConfiguration):
        experiment_directory = configuration.experiment_directory
        updated_experiment_directory = generate_fold_directory(experiment_directory, foldnum)
        configuration.experiment_directory = updated_experiment_directory

    else:
        raise TypeError(f'invalid argument of {type(configuration)}')



def refold_raw_configuration(configuration: dict,
                             strategy: str | CVStrategy,
                             foldnum: int) -> dict:

    if not isinstance(strategy, CVStrategy):
        strategy = CVStrategy(strategy)

    generator_class = strategy_to_generator[strategy]
    generator = generator_class()
    split = generator[foldnum]

    configuration = set_instances_ID(configuration,
                                     training_instances=split['training'],
                                     validation_instances=split['validation'])
    
    update_fold_directory(configuration, foldnum)
    return configuration




def refold_configuration(configuration: dict,
                         strategy: str | CVStrategy,
                         foldnum: int,
                         validate: bool = True) -> dict | TrainingConfiguration:
    """Refold a raw configuration dictionary.

    Select a cross validation strategy and fold number to inject the
    corresponding instance IDs into the configuration.

    Parameters
    ----------

    configuration : dict
        Raw training configuration.

    strategy : str or CVStrategy
        Cross validation strategy identifier.

    foldnum : int
        Fold number. Maximum selectible number depends on internal
        fold generation defaults.

    validate : bool, optional
        Flag to toggle validation of configuration via pydantic model.
        Defaults to True.

    
    Returns
    -------

    refolded_configuration : dict or TrainingConfiguration
        Original configuration with modified instance IDs.
        Return type depends on the selected validation behaviour.
    """

    if not isinstance(strategy, CVStrategy):
        strategy = CVStrategy(strategy)

    generator_class = strategy_to_generator[strategy]
    generator = generator_class()

    if validate:
        configuration = TrainingConfiguration(**configuration)

    split = generator[foldnum]

    configuration = set_instances_ID(configuration,
                                     training_instances=split['training'],
                                     validation_instances=split['validation'])

    update_fold_directory(configuration, foldnum)

    return configuration



def refold_file(source_path: PathLike,
                target_path: PathLike,
                strategy: str | CVStrategy,
                foldnum: int,
                validate: bool = True,
                force_write: bool = False) -> dict | TrainingConfiguration:
    """Refold a YAML configuration file.

    Parameters
    ----------

    source_path : PathLike
        Basal configuration will be read from here.

    target_path : PathLike
        Refolded confgiruation will be stored there.

    strategy : str or CVStrategy
        Cross validation strategy identifier.

    foldnum : int
        Fold number. Maximum selectible number depends on internal
        fold generation defaults.

    validate : bool, optional
        Flag to toggle validation of configuration via pydantic model.
        Defaults to True.

    force_write : bool, optional
        Flag to set overwriting behaviour in case of preexisting file
        at the indicated target location.

        
    Returns
    -------

    refolded_configuration : dict or TrainingConfiguration
    """
    configuration = load_yaml(source_path)

    refolded_configuration = refold_configuration(configuration, strategy, foldnum, validate)

    if validate:
        refolded_configuration = refolded_configuration.model_dump()

    write_yaml(refolded_configuration, target_path, force_write=force_write)

    return refolded_configuration
