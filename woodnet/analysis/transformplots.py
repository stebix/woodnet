"""
Advanced tooling to visualize the parametrized transformations applied to the
dataset instances during inference evaluation.

@jsteb
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import skimage

from collections.abc import Sequence, Mapping
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numbers import Number

from woodnet.inference.parametrized_transforms import ParametrizedTransform



def compute_transform(data: np.ndarray,
                      transform: Sequence[ParametrizedTransform],
                      ) -> tuple[str, dict]:
    """Precompute the transform series and its effects foe arbitrary 2D input data."""
    if data.ndim == 2:
        # add fake batch axis
        preproc_data = torch.from_numpy(data[np.newaxis, ...])
    else:
        preproc_data = torch.from_numpy(data)
                                        
    names = {p.name for p in transform}
    if len(names) != 1:
        raise RuntimeError(f'expected homogenous sequence of ParametrizedTransform, but got names \'{names}\'')
    name = names.pop()
    result = [('stage-0', (), data, {'ssim' : 1.0, 'delta' : 0.0})]
    for i, parametrization in enumerate(transform):
        out = parametrization.transform(preproc_data)
        out = out.detach().squeeze().numpy() if isinstance(out, torch.Tensor) else np.squeeze(out)
        ssim = skimage.metrics.structural_similarity(data, out, data_range=10)
        cum_diff = np.sum(np.abs(out - data))
        result.append(
            (f'stage-{i}', parametrization.parameters, out, {'ssim' : ssim, 'delta' : cum_diff})
        )
    return (name, result)
    


def compute_transforms(data: np.ndarray,
                       transforms: Sequence[Sequence[ParametrizedTransform]],
                       ) -> dict:
    results = {}
    for transform in transforms:
        name, result = compute_transform(data, transform)
        results[name] = result
    return results


def plot_ssim_and_delta(ssim: Sequence[Number], delta: Sequence[Number]) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, ssim_ax = plt.subplots()
    delta_ax = ssim_ax.twinx()
    
    (ssim_line,) = ssim_ax.plot(ssim, label='SSIM', marker='o', color='tab:blue')
    (delta_line,) = delta_ax.plot(delta, label='$\Delta$', marker='o', color='tab:orange')
    # construct single legend
    lines = [ssim_line, delta_line]
    labels = [line.get_label() for line in lines]
    ssim_ax.legend(lines, labels, loc='best')
    ssim_ax.set_ylabel('SSIM')
    delta_ax.set_ylabel(r'Cumulative Difference $\Delta$')
    
    return (fig, (ssim_ax, delta_ax))


def plot_ssim(ssims_mapping: Mapping[str, Sequence[Number]]) -> tuple[Figure, Axes]:
    max_stage_count = max(len(ssims) for ssims in ssims_mapping.values())
    X = range(max_stage_count)
    fig, ax = plt.subplots()
    
    for name, values in ssims_mapping.items():
        x = range(len(values))
        ax.plot(x, values, label=name, marker='o')
    
    ax.set_ylabel('SSIM')
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs=X))
    ax.legend()
    return (fig, ax)


def plot_delta(delta_mapping: Mapping[str, Sequence[Number]]) -> tuple[Figure, Axes]:
    max_stage_count = max(len(deltas) for deltas in delta_mapping.values())
    X = range(max_stage_count)
    fig, ax = plt.subplots()
    
    for name, values in delta_mapping.items():
        x = range(len(values))
        ax.plot(x, values, label=name, marker='o')
    
    ax.set_ylabel('Cumulative Difference $\Delta$')
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs=X))
    ax.legend()
    return (fig, ax)