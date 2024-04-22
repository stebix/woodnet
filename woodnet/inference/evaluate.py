"""
Implements functional interface and predictor class to evaluate the model in the context
of robustness experiments.

High-level overview:

Various functions to evaluate a single model, multiple model instances (checkpoints)
and cross-validation results.

For the first single model evaluation, the model is evaluated with the data contained
in the data loader. Thus, model and data are deemed fixed. For robustness experiments,
the underlying dataset is expected to functionally understand the parametrized transform
attribute that modifies the data yielded. Here, the transformation classes and the
different realizations/parametrizations of a transform class are variable.

For the muliple model evaluation, the core data is again deemed fixed (certain set
of validation data). Here, however, multiple model isntances/cehckpoints are evaluated.
Thusly, a mapping from a unique identifiying ID string to a model object is expected.

For the last fold-wise evaluation, we vary the data per fold since every fold
has its specific set of validation data. Still, every fold has multiple model instances.
This is the reason for the `FoldSpec`, where the fold-wise data loader and model
instances/checkpoints are expected to be bundled together.  
"""
import logging
import dataclasses
import contextlib
from collections.abc import Mapping, Sequence, Callable
from pathlib import Path

import torch
import torch.utils
import tqdm.auto as tqdm

from woodnet.datasets.volumetric_inference import set_parametrized_transform
from woodnet.evaluation.metrics import compute_cardinalities
from woodnet.inference.inference import get_transforms_name
from woodnet.inference.parametrized_transforms import CongruentParametrizations, ParametrizedTransform
from woodnet.trackers import TrackedCardinalities


LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             parametrizations: Sequence[ParametrizedTransform],
             device: torch.device,
             dtype: torch.dtype,
             display_parametrizations_progress: bool,
             display_loader_progress: bool
             ) -> list[dict]:
    """
    Evaluate the model for the given sequence of parametrized transforms.

    Reports the results as dictionary.
    """
    if display_parametrizations_progress:
        settings = {'unit' : 'parametrization', 'desc' : get_transforms_name(parametrizations), 'leave' : False}
        parametrizations = tqdm.tqdm(parametrizations, **settings)

    tracker = TrackedCardinalities()
    # base template for results: list of state dicts for the tracker objects
    results = []
    for parametrization in parametrizations:
        # compute performance of model for every distinct transform
        tracker.reset()
        # set variable transform to current parametrized instance
        set_parametrized_transform(loader.dataset, transform=parametrization)
        wrapped_loader = maybe_wrap_loader(loader, display_loader_progress)

        # improved progress reporting
        if display_parametrizations_progress:
            parametrizations.set_postfix_str(f'parameters = {parametrization.parameters}')

        for batch in wrapped_loader:
            data, label = batch
            data = data.to(device=device, dtype=dtype, non_blocking=True)
            label = label.to(device=device, dtype=dtype, non_blocking=True)
            prediction = model(data)

            cardinalities = compute_cardinalities(prediction, label)
            tracker.update(cardinalities)

        results.append({'parameters' : parametrization.parameters, 'metrics' : tracker.state_dict()})
    return results



class Predictor:

    def __init__(self,
                 model: torch.nn.Module,
                 device: str | torch.device,
                 dtype: torch.dtype,
                 use_amp: bool,
                 use_inference_mode: bool,
                 display_transforms_progress: bool,
                 display_parametrizations_progress: bool,
                 display_loader_progress: bool,
                 leave_transforms_progress: bool = True
                 ) -> None:

        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.use_amp = use_amp
        self.use_inference_mode = use_inference_mode
        self.display_transforms_progress = display_transforms_progress
        self.display_parametrizations_progress = display_parametrizations_progress
        self.display_loader_progress = display_loader_progress
        self.leave_transforms_progress = leave_transforms_progress


    def disabled_gradient_context(self):
        if self.use_inference_mode:
            return torch.inference_mode()
        return torch.no_grad()


    def amp_context(self):
        return torch.autocast(device_type=self.device.type, enabled=self.use_amp)


    def inference_contexts(self) -> tuple:
        """Get configured no-gradient and mixed precision contexts."""
        return (self.disabled_gradient_context(), self.amp_context())


    def configure_model(self) -> None:
        """Jointly adjust model float precision, eval mode and testing flag."""
        self.model.eval()
        logger.debug('Set model to evaluation mode.')
        self.model.testing = True
        final_nonlinearity = getattr(self.model, 'final_nonlinearity', None)
        logger.debug(
            f'Set model testing flag. Final nonlinearity is: {final_nonlinearity}'
        )
        self.model.to(dtype=self.dtype, device=self.device)
        logger.debug(f'Moved model to precision {self.dtype} and device {self.device}')


    def predict(self,
                loader: torch.utils.data.DataLoader,
                transforms: Sequence[CongruentParametrizations]) -> dict:
        """
        Perform full prediction run (full loader) for a number of parmetrizations

        Results are reported as a metadata-enhanced dictionary with
        """
        report = []
        self.configure_model()

        if self.display_transforms_progress:
            settings = {'desc' : 'transformations', 'unit' : 'tf',
                        'leave' : self.leave_transforms_progress}
            transforms = tqdm.tqdm(transforms, **settings)

        with contextlib.ExitStack() as stack:
            # conditionally use amp and inference or no-grad mode for speed/throughput
            [stack.enter_context(cm) for cm in self.inference_contexts()]
            for parametrizations in transforms:
                if self.display_transforms_progress:
                    transforms.set_postfix_str(f'transform=\'{parametrizations.name}\'')

                result = evaluate(self.model,
                                  loader=loader, parametrizations=parametrizations,
                                  device=self.device, dtype=self.dtype,
                                  display_parametrizations_progress=self.display_transforms_progress,
                                  display_loader_progress=self.display_loader_progress)

                parametrization_report = {'metadata' : parametrizations.info(), 'report' : result}
                report.append(parametrization_report)

        return report


from typing import NamedTuple


class TaggedDataElement(NamedTuple):
    """
    Combine basal supervised training data (data, label)
    with parametrized transformation information for robustness testing inference.
    """
    data: torch.Tensor
    label: torch.Tensor
    tag: dict



class AdvancedPredictor:
    eager_configure_model: bool = True

    def __init__(self,
                 ID: str,
                 model: torch.nn.Module,
                 device: str | torch.device,
                 dtype: torch.dtype,
                 use_amp: bool,
                 use_inference_mode: bool,
                 display_transforms_progress: bool,
                 display_parametrizations_progress: bool,
                 display_loader_progress: bool,
                 leave_transforms_progress: bool = True
                 ) -> None:

        self.ID = ID
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.use_amp = use_amp
        self.use_inference_mode = use_inference_mode
        self.display_transforms_progress = display_transforms_progress
        self.display_parametrizations_progress = display_parametrizations_progress
        self.display_loader_progress = display_loader_progress
        self.leave_transforms_progress = leave_transforms_progress

        if self.eager_configure_model:
            self.configure_model()


    def disabled_gradient_context(self):
        if self.use_inference_mode:
            return torch.inference_mode()
        return torch.no_grad()


    def amp_context(self):
        return torch.autocast(device_type=self.device.type, enabled=self.use_amp)


    def inference_contexts(self) -> tuple:
        """Get configured no-gradient and mixed precision contexts."""
        return (self.disabled_gradient_context(), self.amp_context())


    def configure_model(self) -> None:
        """Jointly adjust model float precision, eval mode and testing flag."""
        self.model.eval()
        logger.debug('Set model to evaluation mode.')
        self.model.testing = True
        final_nonlinearity = getattr(self.model, 'final_nonlinearity', None)
        logger.debug(
            f'Set model testing flag. Final nonlinearity is: {final_nonlinearity}'
        )
        self.model.to(dtype=self.dtype, device=self.device)
        logger.debug(f'Moved model to precision {self.dtype} and device {self.device}')


    def predict(self,
                loader: torch.utils.data.DataLoader,
                transforms: Sequence[CongruentParametrizations]) -> dict:
        """
        Perform full prediction run (full loader) for a number of parmetrizations

        Results are reported as a metadata-enhanced dictionary with
        """
        report = []
        self.configure_model()

        if self.display_transforms_progress:
            settings = {'desc' : 'transformations', 'unit' : 'tf',
                        'leave' : self.leave_transforms_progress}
            transforms = tqdm.tqdm(transforms, **settings)

        with contextlib.ExitStack() as stack:
            # conditionally use amp and inference or no-grad mode for speed/throughput
            [stack.enter_context(cm) for cm in self.inference_contexts()]
            for parametrizations in transforms:
                if self.display_transforms_progress:
                    transforms.set_postfix_str(f'transform=\'{parametrizations.name}\'')

                result = evaluate(self.model,
                                  loader=loader, parametrizations=parametrizations,
                                  device=self.device, dtype=self.dtype,
                                  display_parametrizations_progress=self.display_transforms_progress,
                                  display_loader_progress=self.display_loader_progress)

                parametrization_report = {'metadata' : parametrizations.info(), 'report' : result}
                report.append(parametrization_report)

        return report
    


def evaluate_multiple(models: Mapping[str, torch.nn.Module],
                      loader: torch.utils.data.DataLoader,
                      transforms: Sequence[CongruentParametrizations],
                      device: torch.device,
                      dtype: torch.dtype,
                      use_amp: bool,
                      use_inference_mode: bool,
                      display_model_instance_progress: bool,
                      leave_model_instance_progress: bool,
                      display_transforms_progress: bool,
                      display_parametrizations_progress: bool,
                      display_loader_progress: bool
                      ) -> dict:
    """Perform prediction robustness experiment with multiple models.

    Parameters
    ----------

    models : Mapping[str, torch.nn.Module]
        Mapping from model ID string towards instantiated and usable
        model object.

    loader : torch.utils.data.DataLoader
        Usable and configured dataloader instance.
        Underlying dataset should support parametrized transforms:
        robustness experiments are conducted by iteratively in-place setting
        the provided parametrized transforms.

    transforms : Sequence of CongruentParametrizations
        Data-modifying transforms used for robustness testing.

    device : torch.device
        Device used for inference computation.

    dtype : torch.dtype
        Basal dtype model parameters and data is cast to.
        If `use_amp` is selected, certain regions may utilize different data types.

    use_amp : bool
        Flag for automatic mixed precision.

    use_inference_mode : bool
        Flag for inference mode (i.e. an even faster and restrictive veriosn of no_grad)

    display_model_instance_progress : bool
        Use progress bar for checkpoint instances (i.e. top-level loop of this function).

    leave_model_instance_progress : bool
        Leave progress bar in terminal after loop conclusion.

    display_transforms_progress : bool
        Use progress bar for transform class progress.

    display_parametrizations_progress : bool
        Use progress bar for display of different parametrizations of a single transform class.

    display_loader_progress : bool
        Use progress bar for loader progress (batch progress) within a single parametrization
        of a transform class.


    Returns
    -------

    results : dict
        Mapping from model ID string to result dict of the transform-wise evaluation
    """
    if display_model_instance_progress:
        settings = {
            'unit' : 'chkpt', 'leave' : leave_model_instance_progress,
            'desc' : 'checkpoint instances'
        }
        models = tqdm.tqdm(models.items(), **settings)
    else:
        models = models.items()

    results: dict[str, dict] = {}

    for identifier, model in models:
        logger.info(f'Constructing Predictor for model with identifier \'{identifier}\'')
        predictor = Predictor(model=model, device=device,
                              dtype=dtype, use_amp=use_amp,
                              use_inference_mode=use_inference_mode,
                              display_transforms_progress=display_transforms_progress,
                              display_parametrizations_progress=display_parametrizations_progress,
                              display_loader_progress=display_loader_progress,
                              leave_transforms_progress=False)

        result = predictor.predict(loader, transforms=transforms)
        results[identifier] = result

    return results




def check_unique_names(transforms: Sequence[CongruentParametrizations]) -> bool:
    names = {transform.name for transform in transforms}
    return len(names) == len(transforms)

from frozendict import frozendict



def maybe_wrap_transforms(do_wrap: bool,
                          transforms: Sequence[CongruentParametrizations],
                          leave: bool):
    if not do_wrap:
        return transforms
    settings = {
        'unit' : 'tf', 'desc' : 'transforms progress', 'leave' : leave
    }
    return tqdm.tqdm(transforms, **settings)


def maybe_wrap_parametrizations(do_wrap: bool,
                                parametrizations: Sequence[ParametrizedTransform],
                                leave: bool):
    if not do_wrap:
        return parametrizations
    settings = {
        'unit' : 'pts', 'desc' : 'parametrizations progress',
        'leave' : leave
    }
    return tqdm.tqdm(parametrizations, **settings)



def maybe_wrap_loader(do_wrap: bool,
                      loader: torch.utils.data.DataLoader,
                      leave: bool):
    if not do_wrap:
        return loader
    settings = {
        'unit' : 'bt', 'desc' : 'loader progress',
        'leave' : leave
    }
    return tqdm.tqdm(loader, **settings)


def maybe_wrap_models(do_wrap: bool,
                      models: Mapping[str, Callable],
                      leave: bool):
    if not do_wrap:
        return models
    settings = {'unit' : 'mdl', 'desc' : 'model-wise progress', 'leave' : leave}
    return tqdm.tqdm(models, **settings)


def get_inference_contexts(use_inference_mode: bool, use_amp: bool, device: torch.device):
    contexts = []
    if use_inference_mode:
        contexts.append(torch.inference_mode())
    else:
        contexts.append(torch.no_grad())
    contexts.append(torch.autocast(device_type=device.type, enabled=use_amp))
    return contexts


from collections import defaultdict


def prebuild_results_mapping(model_identifiers: Sequence[str],
                             transforms: Sequence[CongruentParametrizations]) -> dict:
    # imperatively pre-build the mapping table
    container = {}
    for model_ID in model_identifiers:
        container[model_ID] = {}
        for transform in transforms:
            container[model_ID][transform.name] = {
                'metadata' : transform.info(),
                'results' : defaultdict(TrackedCardinalities)
            }
    return container


def recursive_to_statedict(d: Mapping) -> dict:
    """
    Recursively transform `TrackedCardinality` instances at any depth of the
    nested instance into their corresponding `state_dict`.
    """
    transformed = {}
    for k, v in d.items():
        if isinstance(v, dict):
            transformed[k] = recursive_to_statedict(v)
        elif isinstance(v, TrackedCardinalities):
            transformed[k] = v.state_dict()
        else:
            transformed[k] = v
    return transformed


def configure_model(model: torch.nn.Module,
                    dtype: torch.dtype,
                    device: torch.device,
                    eval_mode: bool,
                    testing_flag: bool) -> torch.nn.Module:
    """Configure model according to settings."""
    model = model.to(dtype=dtype, device=device)
    if eval_mode:
        model.eval()
    if testing_flag:
        model.testing = True
    return model


class NullProgressBar:

    def update(self):
        pass

    def reset(self):
        pass


def evaluate_multiple_inverted(models: Mapping[str, Callable],
                               loader: torch.utils.data.DataLoader,
                               transforms: Sequence[CongruentParametrizations],
                               dtype: torch.dtype,
                               device: torch.device,
                               use_amp: bool,
                               use_inference_mode: bool,
                               non_blocking_transfer: bool,
                               display_transforms_progress: bool,
                               leave_transforms_progress: bool,
                               display_parametrizations_progress: bool,
                               display_loader_progress: bool,
                               display_models_progress: bool) -> dict:
    """
    Perform robustness evaluations of multiple models with a single data loader
    source over many transforms and parametrizations thereof.

    End result layout:

    model_ID : 
        transform_name : 
            metadata : {metadata_dict}
            results : {
                {parameter : value} : TrackedCardinalities | {'ACC' : value, 'MCC' : value}
            }

    """
    if not check_unique_names(transforms):
        logger.warning('Detected parametrized transforms with non-unique names. '
                       'This will overwrite-reduce computed result metrics.')

    container = prebuild_results_mapping(models.keys(), transforms)
    transforms = maybe_wrap_transforms(do_wrap=display_loader_progress, transforms=transforms,
                                       leave=leave_transforms_progress)    
    contexts = get_inference_contexts(use_inference_mode, use_amp, device)
    # python converts to integer if booleans are added
    # TODO: position computation does not work in jupyter notebook contexts
    if display_models_progress:
        offset = (  display_transforms_progress
                  + display_parametrizations_progress
                  + display_loader_progress)
        models_pbar = tqdm.tqdm(total=len(models), desc='model-wise progress',
                                unit='mdl', position=offset)
    else:
        models_pbar = NullProgressBar()

    # TODO: difficult to understand and rugged left col formatting due to massive
    #       indentation levels: refactor reeeeee
    for transform in transforms:

        if display_transforms_progress:
            transforms.set_postfix_str(f'current=\'{transform.name}\'')

        # Order is important here since the attribute is not reachable after conditional
        # wrapping with the tqdm progress bar utility.
        name = transform.name
        transform = maybe_wrap_parametrizations(do_wrap=display_parametrizations_progress,
                                                parametrizations=transform, leave=False)

        for parametrization in transform:

            if display_parametrizations_progress:
                transform.set_postfix_str(f'current=\'{parametrization.parameters}\'')

            set_parametrized_transform(loader.dataset, parametrization)
            mwrp_loader = maybe_wrap_loader(do_wrap=display_loader_progress, loader=loader,
                                            leave=False)
            for batch in mwrp_loader:

                data, label = batch
                data = data.to(device=device, dtype=dtype, non_blocking=non_blocking_transfer)
                label = label.to(device=device, dtype=dtype, non_blocking=non_blocking_transfer)

                # make predictions with all models for the current data item
                for ID, model in models.items():
                    with contextlib.ExitStack() as stack:
                        [stack.enter_context(ctx) for ctx in contexts]
                        prediction = model(data)

                    cardinalities = compute_cardinalities(prediction, label)

                    # sort into appropriate part of the results mapping
                    parameters = frozendict(parametrization.parameters)
                    tracker = container[ID][name]['results'][parameters]
                    tracker.update(cardinalities)
                    models_pbar.update()
                
                models_pbar.reset()

    return container




from collections import UserDict

class LazyModelDict(UserDict):
    """
    Dict-like container that instantiates models lazily upon request via key.
    
    The model is prouced from the basic ingredients.

    Basic ingredients:
        - Mapping from unique model string ID to checkpoint file location.
        - (top-level) configuration containing the model specification
          e.g. name, kwargs and fully optional compile options
    
    Model object will be newly created from ground up upon value retrieval.
    """
    # Configuration is required but kwarg, maybe design this better.
    def __init__(self, dict=None, /, configuration: Mapping = None, no_compile_override: bool = False) -> None:
        if configuration is None:
            raise TypeError('LazyModelDict requires configuration')
        self._configuration = configuration
        self.no_compile_override = no_compile_override
        super().__init__(dict)
    

    def __setitem__(self, key: str, item: Path) -> None:
        if not isinstance(item, Path):
            logger.warning(f'LazyModelDict expects pathlib.Path values, but got {type(item)}')
        try:
            suffix = item.suffix
        except AttributeError:
            suffix = 'NOTSET'
        
        if not suffix.endswith('pth'):
            logger.warning('Inserted value does have expected \'pth\' suffix.')

        return super().__setitem__(key, item)
    

    def __getitem__(self, key: str) -> Callable | torch.nn.Module:
        """
        Retrieve model item via string ID key.
        Model will be lazily instantiated on the fly.

        NOTE: Currently we create the randomly-init'ed model, compile it and then
              load the trained parameters via the state_dict.
              This may have unforeseen performance consequences. Check this!
        """
        path = super().__getitem__(key)
        logger.debug(f'Requested model instance (ID=\'{key}\') from location \'{path}\'')
        model = create_model(self._configuration, no_compile_override=self.no_compile_override)
        # first force load to CPU/RAM: training and inference device may differ
        state_dict = torch.load(path, map_location='cpu')
        inject_state_dict(model, state_dict)
        logger.debug(f'Successfully re-created model from location: \'{path}\'.')
        return model
        

def create_models_from(pathmap: Mapping[str, Path],
                       configuration: Mapping,
                       dtype: torch.dtype,
                       device: torch.device,
                       no_compile_override: bool = False,
                       eval_mode: bool = True,
                       testing_flag: bool = True
                       ) -> dict[str, Callable | torch.nn.Module]:
    """
    Eagerly resurrect models mapping from a path mapping.
    """
    models = {}
    for ID, path in pathmap.items():
        logger.debug(f'Starting creation of model instance (ID=\'{ID}\''
                     f') from location \'{path}\'')
        model = create_model(configuration, no_compile_override=no_compile_override)
        state_dict = torch.load(path, map_location='cpu')
        inject_state_dict(model, state_dict)
        model = configure_model(model, dtype=dtype, device=device, eval_mode=eval_mode,
                                testing_flag=testing_flag)
        logger.debug(f'Successfully re-created and configured model.')
        models[ID] = model
    return models
        


@dataclasses.dataclass
class DatasetRecipe:
    """
    Recipe with 'batteries included' for the creation of a (validation) dataset.

    Parameters
    ----------

    IDs : Sequence of string
        String ID of the dataset instances
    
    class_name : str
        Dataset class name
    
    transform_confs : Sequence of Mapping
        Transformation configurations as sequence of Mapping. Here we mean the
        static transforms that remain unchanged for the lifetime of
        the dataset. In the validation/testing context, this is usually
        the normalization.

    kwargs : Mapping
        Any further kwargs necessary for dataset creation.
        Usual choices are e.g. 'tileshape' for volumetric datasets or
        'axis' for planar datasets.
    """
    IDs: Sequence[str]
    transform_confs: Sequence[dict]
    class_name: str
    kwargs: dict


from woodnet.configtools.validation import TrainingConfiguration

def deduce_dataset_recipe(configuration: Mapping) -> DatasetRecipe:
    """
    Deduce all necessary dataset buikder ingredients, i.e. the `DatasetRecipe`
    from the top-level training configuration.
    """
    conf = TrainingConfiguration(**configuration)
    # retrieve values from the validate structure
    class_name = conf.loaders.dataset
    instances_ID = conf.loaders.val.instances_ID
    # hopefully this produces a copy!?
    transform_configurations = [
        elem.model_dump() for elem in conf.loaders.val.transform_configurations
    ]
    kwargs = conf.loaders.dataset_kwargs()
    return DatasetRecipe(IDs=instances_ID, class_name=class_name,
                         transform_confs=transform_configurations,
                         kwargs=kwargs)



@dataclasses.dataclass
class EvalSpec:
    """
    Encapsulate the core ingredients for the evaluation of a
    training result:

    - the mapping from unique ID string(s) to model objects(s)
      (all registered checkpoints will be evaluated)

    - the validation data ID strings
      that were utilized for the respective CV fold
    """
    models: Mapping[str, Path]
    dataset_recipe: DatasetRecipe



def evaluate_folds(evalspecs: dict[str, EvalSpec],
                   transforms: Sequence[CongruentParametrizations],
                   device: torch.device,
                   dtype: torch.dtype,
                   use_amp: bool,
                   use_inference_mode: bool,
                   display_fold_progress: bool,
                   leave_fold_progress: bool,
                   display_model_instance_progress: bool,
                   display_transforms_progress: bool,
                   display_parametrizations_progress: bool,
                   display_loader_progress: bool
                   ) -> dict:
    """
    Run a full prediction/evaluation over arbitrarily many folds.

    Deeply nested prediction that combines loops over the following elements:

    {fold-1, ..., fold-N}
        {checkpoint-1, ..., checkpoint-N}
            {transform-1, ..., transform-N}
                {parametrization-1, ..., parametrization-N}
                    dataloader
    """
    if display_fold_progress:
        settings = {'unit' : 'fold', 'desc' : 'fold-wise inference',
                    'leave' : leave_fold_progress}
        evalspecs = tqdm.tqdm(evalspecs.items(), **settings)
    else:
        evalspecs = evalspecs.items()

    results: dict[str, dict] = {'folds' : {}}

    for fold_ID, foldspec in evalspecs:
        result = evaluate_multiple(models=foldspec.models_mapping,
                                   loader=foldspec.loader,
                                   transforms=transforms,
                                   device=device, dtype=dtype, use_amp=use_amp,
                                   use_inference_mode=use_inference_mode,
                                   display_model_instance_progress=display_model_instance_progress,
                                   display_transforms_progress=display_transforms_progress,
                                   display_parametrizations_progress=display_parametrizations_progress,
                                   display_loader_progress=display_loader_progress)

        result['folds'][fold_ID] = result

    return results




from woodnet.inference.directories import TrainingResultBag
from woodnet.inference.inference import (extract_IDs, extract_model_config,
                                         transmogrify_state_dict,
                                         deduce_loader_from_training)
from woodnet.inference.utils import parse_checkpoint

from woodnet.models import get_model_class


def inject_state_dict(model: torch.nn.Module, state_dict: Mapping) -> None:
    """
    Loads state dict into the given model.
    Helper function that automatically handles state dicts reconstructed
    from compiled models.
    """
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        logger.warning('Direct state dict loading failed with runtime error. Re-attempting '
                       'with transmogrified state dict.')
    
        state_dict = transmogrify_state_dict(state_dict)
        model.load_state_dict(state_dict)
    logger.info('Successfully loaded state dictionary.')


from woodnet.models import create_model

def _legacy_produce_evalspec_from(training_results_bag: TrainingResultBag) -> EvalSpec:
    """
    Produce the `EvalSpec` instance from the training results bag.

    Parameters
    ----------

    training_results_bag : TrainingResultsBag
        Compound data structure of training results.
    """
    configuration = training_results_bag.fetch_configuration()
    IDs = extract_IDs(configuration)

    models: dict[str, torch.nn.Module] = {}

    for chkpt in training_results_bag.registered_checkpoints:
        # deduce model template/class from the training configuration and
        # create usable instances. Possibly compiles if configuration defines it. 
        model = create_model(configuration)
        # first force load to CPU/RAM: training and inference device may differ
        state_dict = torch.load(chkpt.path, map_location='cpu')
        logger.info(f'Successfully loaded checkpoint state dict from location \'{chkpt.path}\'.')
        inject_state_dict(model, state_dict)
        # construct model ID string: $qualifier_$UUID  or $UUID depdening on qualifier value
        identifier = '_'.join((chkpt.qualifier, chkpt.UUID)) if chkpt.qualifier else chkpt.UUID
        models[identifier] = model

    return EvalSpec(models, IDs)



def produce_evalspec_from(training_results_bag: TrainingResultBag) -> EvalSpec:
    """
    Produce the `EvalSpec` instance from the training results bag.

    Parameters
    ----------

    training_results_bag : TrainingResultsBag
        Compound data structure of training results.
    """
    configuration = training_results_bag.fetch_configuration()
    recipe = deduce_dataset_recipe(configuration)
    # build mapping: string identifier to model parameters path
    models = {}
    for chkpt in training_results_bag.registered_checkpoints:
        model_ID = chkpt.make_ID()
        models[model_ID] = chkpt.path
    return EvalSpec(models=models, dataset_recipe=recipe)


from woodnet.inference.directories import CrossValidationResultsBag


def produce_foldspec_from(cv_results_bag: CrossValidationResultsBag) -> dict[str, EvalSpec]:
    foldspec = {}
    for foldnum, training_results_bag in cv_results_bag.folds.items():
        foldspec[f'fold-{foldnum}'] = produce_evalspec_from(training_results_bag)
    return foldspec

