"""
Provide preliminary tooling to test the model.

@Author: Jannik Stebani 2024
"""
from collections.abc import Mapping

import torch
import pandas as pd

import woodnet.analysis.dataframetools as dftools
from woodnet.inference.evaluate import evaluate_multiple_inverted, recursive_value_to_statedict
from woodnet.inference.parametrized_transforms import CongruentTransformList, ParametrizedTransform


def quick_test(models: Mapping[str, torch.nn.Module],
               loader: torch.utils.data.DataLoader,
               device: str = 'cuda:0',
               dtype: torch.dtype = torch.float32,
               **kwargs
               ) -> pd.DataFrame:
    """
    Preliminary test-set styled evaluation of the model.
    Runs prediction and metric-wise evaluation on the given loader with
    no test-time data alterations.
    """
    base_kwargs = {
        'transforms': [
            CongruentTransformList([ParametrizedTransform.make_identity()])
        ],
        'dtype': dtype, 'device': torch.device(device),
        'use_amp': True, 'use_inference_mode': True, 'display_transforms_progress': True,
        'display_loader_progress': True, 'display_parametrizations_progress': True,
        'non_blocking_transfer': True, 'leave_transforms_progress': True, 'display_models_progress': True
    }
    kwargs = {'models': models, 'loader': loader} | base_kwargs | kwargs
    result = evaluate_multiple_inverted(**kwargs)
    result_primitive = recursive_value_to_statedict(result)
    df = dftools.preprocess_evaluation_result({'fold-1' : result_primitive})
    return df