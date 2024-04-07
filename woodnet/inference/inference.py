"""
Facilitate inference runs.


@jsteb 2024
"""
import torch

from woodnet.inference.loaders import generate_loader
from woodnet.inference.parametrized_transforms import (ParametrizedTransform,
                                                       maybe_wrap_parametrized_transforms,
                                                       CoherentList)
from woodnet.datasets.volumetric_inference import InferenceTileDataset
from woodnet.trackers import TrackedCardinalities
from woodnet.evaluation.metrics import compute_cardinalities

"""
Desired result dictionary template for evaluate run with single
sequence of parametrized transform.

mydict = {
    'metadata' : {
        'name' : GaussianSmooth,
        'transform_class_name' : 'app.module.GaussianSmooth',
        'parameter_set' : {'sigma'}
    },
    'metrics' : [
        {'parameters' : {'sigma' : 1}, 'values' : {'ACC' : 1.0, 'MCC' : 0.8}},
        {'parameters' : {'sigma' : 2}, 'values' : {'ACC' : 0.9, 'MCC' : 0.77}}        
    ]
}
"""

def evaluate(model: torch.nn.Module,
             parametrized_transforms: CoherentList[ParametrizedTransform],
             dataset: InferenceTileDataset,
             batch_size: int,
             device: str | torch.device,
             dtype: torch.dtype = torch.float32
            ) -> dict:
    """
    Evaluate the model for the given sequence of parametrized transforms.

    Reports the results as dictionary.
    """
    device = torch.device(device) if isinstance(device, str) else device
    loader = generate_loader(dataset, batch_size)
    parametrized_transforms = maybe_wrap_parametrized_transforms(parametrized_transforms)
    tracker = TrackedCardinalities()
    # set to evaluation/testing mode: freeze some layer parameters (batch norm, dropout, ...)
    # and apply final nonlinearity if defined on model object
    model.eval()
    model.testing = True
    
    results = {'metadata' : parametrized_transforms.info(), 'metrics' : []}
    
    for transform in parametrized_transforms:
        # set variable transform to current parametrized instance
        loader.dataset.parametrized_transform = transform
        tracker.reset()
        # improved progress reporting
        try:
            parametrized_transforms.set_postfix_str(f'current_parameters = {transform.parameters}')
        except AttributeError:
            pass
        
        for batch in loader:
            data, label = batch
            data = data.to(device=device, dtype=dtype, non_blocking=True)
            label = label.to(device=device, dtype=dtype, non_blocking=True)
            prediction = model(data)
            
            cardinalities = compute_cardinalities(prediction, label)
            tracker.update(cardinalities)
            
        results['metrics'].append({'parameters' : transform.parameters, 'values' : tracker.state_dict()})
    return results