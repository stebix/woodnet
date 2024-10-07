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
from collections import defaultdict
from collections.abc import Mapping, Sequence, Callable
from pathlib import Path
from typing import NamedTuple

import torch
import torch.utils
import tqdm.auto as tqdm
from frozendict import frozendict

from woodnet.datasets.volumetric_inference import set_parametrized_transform
from woodnet.evaluation.metrics import compute_cardinalities
from woodnet.inference.parametrized_transforms import CongruentParametrizations, ParametrizedTransform
from woodnet.trackers import TrackedCardinalities
from woodnet.inference.directories import TrainingResultBag, CrossValidationResultsBag
from woodnet.configtools.validation import TrainingConfiguration
from woodnet.inference.resurrection import resurrect_models_from
from woodnet.transformations.transforms import ToDevice

ContextManager = contextlib.AbstractContextManager

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def get_transforms_name(transforms) -> str:
    """Try to get/deduce the global semantical name of the transforms."""
    try:
        name = transforms.name
    except AttributeError:
        name = 'parametrized transforms'
    return name


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
        wrapped_loader = maybe_wrap_loader(display_loader_progress, loader, False)

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


    def disabled_gradient_context(self) -> ContextManager:
        if self.use_inference_mode:
            return torch.inference_mode()
        return torch.no_grad()


    def amp_context(self) -> ContextManager:
        return torch.autocast(device_type=self.device.type, enabled=self.use_amp)


    def inference_contexts(self) -> tuple[ContextManager, ContextManager]:
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


class TaggedDataElement(NamedTuple):
    """
    Combine basal supervised training data (data, label)
    with parametrized transformation information for robustness testing inference.
    """
    data: torch.Tensor
    label: torch.Tensor
    tag: dict


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


def get_inference_contexts(use_inference_mode: bool, use_amp: bool,
                           device: torch.device) -> list[ContextManager]:
    contexts = []
    if use_inference_mode:
        contexts.append(torch.inference_mode())
    else:
        contexts.append(torch.no_grad())
    contexts.append(torch.autocast(device_type=device.type, enabled=use_amp))
    return contexts


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


def recursive_value_to_statedict(d: Mapping) -> dict:
    """
    Recursively transforms `TrackedCardinality` instances selectively
    (only Mapping values) at any depth of the (possibly nested)
    mapping into their corresponding state dictionary via
    the corresponding method. 
    """
    transformed = {}
    for k, v in d.items():
        if isinstance(v, dict):
            transformed[k] = recursive_value_to_statedict(v)
        elif isinstance(v, TrackedCardinalities):
            transformed[k] = v.state_dict()
        else:
            transformed[k] = v
    return transformed


def recursive_key_to_string(d: Mapping) -> dict:
    """
    Recursively transform `frozendict` instances selectively 
    (only Mapping keys) at any depth of the
    (possibly nested) mapping into their corresponding 
    string representation. 
    """
    transformed = {}
    for k, v in d.items():
        if isinstance(k, frozendict):
            k = str(k)
        if isinstance(v, dict):
            transformed[k] = recursive_key_to_string(v)
        else:
            transformed[k] = v
    return transformed


class NullProgressBar:
    """Minimal stand-in for tqdm progress bar object."""
    def update(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass

    def __str__(self) -> str:
        return 'NullProgressBar()'


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
                       'This is probably unwanted and will overwrite some '
                       'computed result metrics.')
        
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

    models_pbar.close()
    return container



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
        Usual options are e.g. 'tileshape' for volumetric datasets or
        'axis' for planar datasets.
    """
    IDs: Sequence[str]
    transform_confs: Sequence[dict]
    class_name: str
    kwargs: dict



def deduce_dataset_recipe(configuration: Mapping) -> DatasetRecipe:
    """
    Deduce all necessary dataset builder ingredients, i.e. the `DatasetRecipe`
    from the top-level training configuration.
    """
    conf = TrainingConfiguration(**configuration)
    # retrieve values from the validate structure
    class_name = conf.loaders.dataset
    instances_ID = conf.loaders.val.instances_ID
    # hopefully this produces a copy!?
    # we use the validation transform configuration and expect here only
    # fixed, non-augmenting transforms such as static scaling are defined
    transform_configurations = [
        elem.model_dump() for elem in conf.loaders.val.transform_configurations
    ]
    kwargs = conf.loaders.dataset_kwargs()
    return DatasetRecipe(IDs=instances_ID, class_name=class_name,
                         transform_confs=transform_configurations,
                         kwargs=kwargs)


from woodnet.datasets import get_builder_class


def create_loader_from(recipe: DatasetRecipe,
                       batch_size: int,
                       num_workers: int,
                       shuffle: bool = False,
                       pin_memory: bool = False
                       ) -> torch.utils.data.DataLoader:
    builder_class = get_builder_class(recipe.class_name)
    builder = builder_class()
    datasets = builder.build(phase='val',
                             instances_ID=recipe.IDs,
                             transform_configurations=recipe.transform_confs,
                             **recipe.kwargs)
    dataset = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=num_workers, shuffle=shuffle,
                                         pin_memory=pin_memory)
    return loader



@dataclasses.dataclass
class EvalSpec:
    """
    Encapsulate the core ingredients for the evaluation of the results
    of a single training experiment:

    - the mapping from unique model ID string(s) to corresponding
      trained parameter files of the model objects(s) on disk

    - the dataset recipe from which it can be fully resurrected
    """
    models_pathmap: Mapping[str, Path]
    dataset_recipe: DatasetRecipe
    configuration: dict


def inject_to_device_transform(loader: torch.utils.data.DataLoader,
                               device: torch.device,
                               non_blocking: bool = True):
    """
    Hack to insert to-device moving function into the
    dataset transformer.
    Note: Use with care! After this, all subsequent tensor transforms
    have to support on-device execution!
    """
    to_device_transform = ToDevice(device=device, non_blocking=non_blocking)
    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        for subdataset in dataset.datasets:
            subdataset.transformer.transforms.insert(0, to_device_transform)
    else:
        dataset.transformer.transforms.insert(0, to_device_transform)
    logger.info(f'Injected {to_device_transform} into DataLoader')


def evaluate_folds(evalspecs: Mapping[str, EvalSpec],
                   transforms: Sequence[CongruentParametrizations],
                   batch_size: int,
                   device: torch.device,
                   dtype: torch.dtype,
                   use_amp: bool,
                   use_inference_mode: bool,
                   non_blocking_transfer: bool = True,
                   num_workers: int = 0,
                   shuffle: int = False,
                   pin_memory: bool =False,
                   no_compile_override: bool = False,
                   display_fold_progress: bool = True,
                   leave_fold_progress: bool = True,
                   display_models_progress: bool= True,
                   display_transforms_progress: bool= True,
                   display_parametrizations_progress: bool= True,
                   display_loader_progress: bool = True,
                   _inject_early_to_device: bool = True
                   ) -> dict:
    """
    Run a full prediction/evaluation over arbitrarily many folds.

    Parameters
    ----------

    evalspecs : Mapping[str, EvalSpec]
        Mapping from fold-wise string ID to evaluation specification that holds
        model template and parameter information and the dataset information.
    
    transforms : Sequence of CongruentParametrizations
        Sequence of containers for parametrized transforms that are applied to
        the data elements. Utilized for robustness experiments.

    batch_size : int
        Batch sizse for the data loader.

    device : torch.device
        Inference device.
        Note: All models are pushed there aggressively and eagerly. Use
        `LazyModelDict` with garbage collections if OOM errors occur.

    dtype : torch.dtype
        Basal data type for model and data.

    use_amp : bool
        Use automatic mixed precision for core model forward method.

    use_inference_mode : bool
        Use PyTorch `inference_mode` instead of default `no_grad` mode.

    non_blocking_transfer : bool, optional
        Use non-blocking transfer when moving data to set device.
        Defaults to `True`.

    num_workers : int, optional
        Number of worker processes used by the PyTorch `DataLoader`.
        Defaults to 0, i.e. loading and processing in main process.

    shuffle : bool, optional
        Shuffle the data elements for repeated iterations over
        the loader. Defaults to `False`.

    pin_memory : bool, optional
        Pin memory setting of the `DataLoader`. Defaults to `False`.

    no_compile_override : bool, optional
        Flag to disable model compilation during the resurrection process.
        If not set, the compilation options from the training configuration
        are used. Defaults to `False`.

    display_fold_progress : bool optional
        Draw progress bar for fold-wise evaluation progress. Defaults to `True`.

    leave_fold_progress : bool, optional
        Keep fold-wise progress bar after loop end. Defaults to `True`.

    display_models_progress : bool optional
        Draw progress bar for model-instance-wise evaluation progress. Defaults to `True`.

    display_transforms_progress : bool optional
        Draw progress bar for transforms-wise evaluation progress. Defaults to `True`.

    display_parametrizations_progress : bool optional
        Draw progress bar for parametrizations-wise evaluation progress. Defaults to `True`.

    display_loader_progress : bool optional
        Draw progress bar for loader-wise evaluation progress. Defaults to `True`.

    _inject_early_to_device : bool, optional
        Inject `ToDevice` transform into the static transforms attribute list
        of the dataset of the loader.
        NOTE: Use with care! This pushes all tensors on the device as first
        action of the transformer pipeline!
        This means that all subsequent transforms must be able to run with
        device-located tensors! Furthermore, this may lead to OOM errors.

    Notes
    -----

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

    # TODO: Solve this more elegantly
    # using num_worker > 0 and CUDA tensors in data loading leads to:
    # RuntimeError: Cannot re-initialize CUDA in forked subprocess.
    # To use CUDA with multiprocessing, you must use the 'spawn' start method
    if _inject_early_to_device and num_workers > 0:
        logger.warning(
            f'Early to-device transfer and multiprocessing data loading is '
            f'not supported. Overriding incompatible setting {num_workers=} '
            f'to compatible setting num_workers = 0'
        )
        num_workers = 0

    for fold_ID, evalspec in evalspecs:
        # preparation setup for this specific fold
        models = resurrect_models_from(evalspec.models_pathmap,
                                       configuration=evalspec.configuration,
                                       dtype=dtype, device=device,
                                       no_compile_override=no_compile_override,
                                       eval_mode=True, testing_flag=True)

        loader = create_loader_from(recipe=evalspec.dataset_recipe,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=shuffle,
                                    pin_memory=pin_memory)
        
        if _inject_early_to_device:
            inject_to_device_transform(loader, device=device,
                                       non_blocking=non_blocking_transfer)

        result = evaluate_multiple_inverted(models=models,
                                            loader=loader,
                                            transforms=transforms,
                                            device=device, dtype=dtype, use_amp=use_amp,
                                            use_inference_mode=use_inference_mode,
                                            non_blocking_transfer=non_blocking_transfer,
                                            display_transforms_progress=display_transforms_progress,
                                            leave_transforms_progress=False,
                                            display_parametrizations_progress=display_parametrizations_progress,
                                            display_loader_progress=display_loader_progress,
                                            display_models_progress=display_models_progress)

        results['folds'][fold_ID] = recursive_value_to_statedict(result)

    return results



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
    models_pathmap = {}
    for chkpt in training_results_bag.registered_checkpoints:
        model_ID = chkpt.make_ID()
        models_pathmap[model_ID] = chkpt.path
    return EvalSpec(models_pathmap=models_pathmap, dataset_recipe=recipe,
                    configuration=configuration)



def produce_foldspec_from(cv_results_bag: CrossValidationResultsBag) -> dict[str, EvalSpec]:
    foldspec = {}
    for foldnum, training_results_bag in cv_results_bag.folds.items():
        foldspec[f'fold-{foldnum}'] = produce_evalspec_from(training_results_bag)
    return foldspec

