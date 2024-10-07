"""
Advanced plotting tooling for visualization of prepared results dataframes.

@jsteb 2024
"""
import numpy as np
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.figure import Figure
from matplotlib.axes import Axes


def dual_performance_plot(df: pd.DataFrame, metric: str,
                          noise_likes: Sequence[str] | None = None,
                          resol_likes: Sequence[str] | None = None,
                          metric_alias: str | None = None) -> tuple[Figure, Sequence[Axes]]:
    """
    Plot the accuracy as two-panel plot for a properly prepared dataframe.
    The dataframe should have a multi-index with stage-0 set for every transform.
    """
    df = df.reset_index()
    df = df.loc[df.metric == metric]
    # this will probably break for differently structured data frames
    df.columns = ('fold', 'model_ID', 'transform', 'stage', 'metric', 'value')
    noise_transforms: Sequence[str] = noise_likes or ['GaussianNoise', 'GibbsNoise', 'PoissonNoise']
    resolution_transforms: Sequence[str] = resol_likes or ['GaussianSmooth', 'LowResolution', 'Zoom']
    # note: direct dot access to `.transform` does not work due to shadowing with transform method
    subdf_noise = df.loc[df['transform'].isin(noise_transforms)] 
    subdf_resol = df.loc[df['transform'].isin(resolution_transforms)]
    
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 3))
    axes = axes.flat
    
    ax = axes[0]
    sn.lineplot(subdf_noise, x='stage', y='value', hue='transform', style='transform',
                errorbar='sd', ax=ax, legend='auto', markers=True)
    # set xaxis ticks
    noise_stages = len(np.unique(subdf_noise.stage))
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(noise_stages)))
    ax.set_xticklabels(range(noise_stages))
    ylabel = metric_alias or metric
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower left')

    ax = axes[1]
    sn.lineplot(subdf_resol, x='stage', y='value', hue='transform', style='transform',
                errorbar='sd', ax=ax, legend='auto', markers=True)
    # legend
    ax.legend(loc='lower left')
    # set xaxis ticks
    resol_stages = len(np.unique(subdf_resol.stage))
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(resol_stages)))
    ax.set_xticklabels(range(resol_stages))

    for ax in axes:
        ax.set_xlabel('Stage Index')
        
    return (fig, axes)


def savefig(fig: Figure,
            filename: str,
            directory: str | Path | None = None,
            dpi: int | None = None,
            bbox_inches: str | None = 'tight',
            **kwargs) -> Path:
    """Save a `matplotlib.figure.Figure` in a programmatic fashion."""
    if directory is None:
        directory = Path('.')
        
    if not directory.is_dir():
        if directory.is_file():
            raise FileExistsError(f'can not use tentative save directory \'{directory}\': is a file')
        elif not directory.exists():
            directory.mkdir()
        else:
            raise OSError(f'wierd preexisting file system object at \'{directory}\' prevents figure save process')
        
    save_path = directory / filename
    if save_path.is_file():
        raise FileExistsError(f'cannot write to \'{save_path}\': file already exists')
    
    if dpi:
        kwargs['dpi'] = dpi
    if bbox_inches:
        kwargs['bbox_inches'] = bbox_inches
        
    fig.savefig(save_path, **kwargs)
    return save_path
    