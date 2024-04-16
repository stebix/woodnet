"""
Implements functional interface to evaluate the model in the context of robustness experiments.

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
from collections.abc import Mapping, Sequence

import torch
import torch.utils
import tqdm

from woodnet.datasets.volumetric_inference import set_parametrized_transform
from woodnet.evaluation.metrics import compute_cardinalities
from woodnet.inference.inference import get_transforms_name, maybe_wrap_loader
from woodnet.inference.parametrized_transforms import CongruentParametrizations, ParametrizedTransform
from woodnet.inference.predictor import Predictor
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



def evaluate_multiple(models_mapping: Mapping[str, torch.nn.Module],
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

    models_mapping : Mapping[str, torch.nn.Module]
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
        models_mapping = tqdm.tqdm(models_mapping.items(), **settings)
    else:
        models_mapping = models_mapping.items()

    results: dict[str, dict] = {}

    for identifier, model in models_mapping:
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



@dataclasses.dataclass
class FoldSpec:
    """
    Encapuslate the core ingredients for evaluating of fold-wise training
    result:

    - the mapping from unique ID string(s) to model file(s)
      (all checkpoints will be evaluated)

    - the data loader instance holding the validation dataset(s)
      that were utilized for the respective CV fold

    The underlying dataset class of the data loader should support
    `parametrized_transforms`.
    """
    models_mapping: dict[str, torch.nn.Module]
    loader: torch.utils.data.DataLoader



def evaluate_folds(foldspecs: dict[int, FoldSpec],
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
        foldspecs = tqdm.tqdm(foldspecs.items(), **settings)
    else:
        foldspecs = foldspecs.items()

    results: dict[str, dict] = {'folds' : {}}

    for foldnum, foldspec in foldspecs:
        result = evaluate_multiple(models_mapping=foldspec.models_mapping,
                                   loader=foldspec.loader,
                                   transforms=transforms,
                                   device=device, dtype=dtype, use_amp=use_amp,
                                   use_inference_mode=use_inference_mode,
                                   display_model_instance_progress=display_model_instance_progress,
                                   display_transforms_progress=display_transforms_progress,
                                   display_parametrizations_progress=display_parametrizations_progress,
                                   display_loader_progress=display_loader_progress)

        result['folds'][foldnum] = result

    return results
