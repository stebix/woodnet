"""
Implements extraction and concatenation of parameters, i.e. weights
and corresponding gradients from ResNet-like models.

2024 @jsteb
"""
import re
import warnings
import numpy as np
import torch

from itertools import product
from collections import defaultdict

from woodnet.models.buildingblocks import ResNetBlock


def is_homogenous(iterable, class_or_tuple) -> bool:
    return all(isinstance(elem, class_or_tuple) for elem in iterable)


def recursive_concatenate(d: dict) -> None:
    for key, value in d.items():
        if is_homogenous(value, np.ndarray):
            d[key] = np.concatenate(
                [t.flatten() for t in value]
            )
        elif isinstance(value, dict):
            recursive_concatenate(value)
        else:
            pass
    return None
            
        
def extract_weigths(name: str, module: torch.nn.Module) -> dict:
    weights = defaultdict(dict)
    module_type, index = name.split('_')
    
    if module_type not in {'conv', 'bn'}:
        warnings.warn(f'extracting weights from unusually named module \'{name}\'')
        
    weight_types = ('weight', 'bias')
    for weight_type in weight_types:
        values = getattr(module, weight_type)
        if values is not None:
            weights[module_type][weight_type] = values.detach().cpu().numpy()
    return weights


def extract_gradients(name: str, module: torch.nn.Module) -> dict:
    gradients = defaultdict(dict)
    module_type, index = name.split('_')
    
    if module_type not in {'conv', 'bn'}:
        warnings.warn(f'extracting weights from unusually named module \'{name}\'')
        
    weight_types = ('weight', 'bias')
    for weight_type in weight_types:
        values = getattr(module, weight_type)
        if values is not None:
            gradients[module_type][weight_type] = values.grad.detach().cpu().numpy()
    return gradients


def extract_weights_from_ResNetBlock(rnblk: ResNetBlock) -> dict[str, np.ndarray]:
    weightvectors = defaultdict(lambda: defaultdict(list))
    names = ('conv', 'bn')
    indices = (1, 2)
    types = ('weight', 'bias')

    for name, index, type in product(names, indices, types):
        module_name = f'{name}_{index}'
        module = getattr(rnblk, module_name)
        values = getattr(module, type)
        # skip appending empty/None values, e.g. for non-bias-carrying norm layers
        if values is not None:
            weightvectors[name][type].append(
                values.detach().cpu().numpy()
            )
    return weightvectors


def extract_gradients_from_ResNetBlock(rnblk: ResNetBlock) -> dict[str, np.ndarray]:
    gradientvectors = defaultdict(lambda: defaultdict(list))
    names = ('conv', 'bn')
    indices = (1, 2)
    types = ('weight', 'bias')

    for name, index, type in product(names, indices, types):
        module_name = f'{name}_{index}'
        module = getattr(rnblk, module_name)
        values = getattr(module, type)
        # skip appending empty/None values, e.g. for non-bias-carrying norm layers
        if values is not None:
            gradientvectors[name][type].append(
                values.grad.detach().cpu().numpy()
            )
    return gradientvectors



def export_parameters(p: torch.Tensor) -> np.ndarray:
    return p.detach().cpu().numpy()

layer_pattern = re.compile('^layer_[0-9]$')

def is_simple_resnet_layer(name: str, module: torch.nn.Module) -> bool:
    """Soft precheck to decide whether named child is simple ResNet layer."""
    if layer_pattern.match(name) is None:
        return False
    if not isinstance(module, torch.nn.Sequential):
        return False
    return True
    

def extract_simple_resnet_parameters(model: torch.nn.Module) -> tuple[dict, list[str]]:
    skipped_children: list[str] = []
    parameters = defaultdict(lambda: defaultdict(dict))
    for name, child in model.named_children():
        
        if is_simple_resnet_layer(name, child):
            for index, subblock in child.named_children():
                parameters[name][index] = extract_weights_from_ResNetBlock(subblock)
        
        elif name in {'conv_1', 'bn_1'}:
            parameters['layer_0']['0'].update(extract_weigths(name, child))
        
        else:
            skipped_children.append(name)
    
    return (parameters, skipped_children)



def extract_simple_resnet_gradients(model: torch.nn.Module) -> tuple[dict, list[str]]:
    skipped_children: list[str] = []
    gradients = defaultdict(lambda: defaultdict(dict))
    for name, child in model.named_children():
        
        if is_simple_resnet_layer(name, child):
            for index, subblock in child.named_children():
                gradients[name][index] = extract_gradients_from_ResNetBlock(subblock)
        
        elif name in {'conv_1', 'bn_1'}:
            gradients['layer_0']['0'].update(extract_gradients(name, child))
        
        else:
            skipped_children.append(name)
    
    return (gradients, skipped_children)



def convert_to_flat(d: dict) -> dict[str, np.ndarray]:
    """
    Convert nested layer-specifying dictionary to flat
    tag-to-parameter array dictionary.
    """
    flat = {}
    for layername, sublayermapping in d.items():
        for sublayer_ID, sublayer in sublayermapping.items():
            for opname, valuemapping in sublayer.items():
                for valuename, valuearray in valuemapping.items():
                    tag = f'{layername}/sublayer_{sublayer_ID}/{opname}/{valuename}'
                    if isinstance(valuearray, (list, tuple)):
                        valuearray = np.concatenate([w.flatten() for w in valuearray], axis=0)
                    else:
                        valuearray = valuearray.flatten()
                    flat[tag] = valuearray.flatten()
    return flat
