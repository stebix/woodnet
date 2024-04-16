"""
Perform a realistic inference benchmark that measures the
per-batch times for various settings of batch size.
"""
import logging
import random
import time
import json

from collections.abc import Sequence
from pathlib import Path

import rich
import torch
import tqdm

from torch.utils.data import ConcatDataset


from woodnet.inference.evaluate import evaluate
from woodnet.models.volumetric import ResNet3D
from woodnet.inference.parametrized_transforms import (ParametrizedTransform,
                                                       generate_parametrized_transforms)
from woodnet.datasets.volumetric_inference import TransformedTileDatasetBuilder
from woodnet.datasets.volumetric_inference import set_parametrized_transform
from woodnet.utils import create_timestamp

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def is_odd(x) -> bool:
    return x % 2 != 0


def make_smoother() -> ParametrizedTransform:
    """
    Create single, rather compute-heavy parametrized transform.
    """
    smoother_specification = {'name' : 'GaussianSmooth', 'parameters' : {'sigma' : 3.0}}
    smoother = generate_parametrized_transforms(smoother_specification, squeeze=False)
    return smoother


def make_dataset(tileshape: tuple[int, int, int],
                 size: int | None) -> ConcatDataset:
    """
    Create volumetric dataset with single normalizing transform.

    Parameters
    ----------

    tileshape
        3D volume chunk shape

    size : int
        Total number of elements in the resulting dataset.
        Use sensible numbers, basal data source are two
        wood volume cylinders.
    """
    transform_configurations = [{'name' : 'Normalize3D', 'mean' : 110, 'std' : 950}]
    ID: list[str] = ['CT10', 'CT9'] 
    builder = TransformedTileDatasetBuilder()
    datasets = builder.build(instances_ID=ID, tileshape=tileshape,
                             transform_configurations=transform_configurations)
    if size:
        if is_odd(size):
            n_first = size // 2
            n_second = n_first + 1
        else:
            n_first = size // 2
            n_second = n_first

        subsamples = [random.sample(range(n)) for n in (n_first, n_second)]
        datasets = [
            torch.utils.data.Subset(dataset, indices)
            for dataset, indices in zip(datasets, subsamples, strict=True)
        ]

    assert len(datasets) == 2, 'setup failure, expecting two dataset instances'
    dataset = ConcatDataset(datasets)
    if size:
        assert len(dataset) == size, (f'faulty subset selection: total '
                                      f'length = {len(dataset)} and size = {size}')
    return dataset


def make_loader(tileshape: tuple[int, int, int],
                batch_size: int,
                num_workers: int,
                shuffle: bool = False) -> torch.utils.data.DataLoader:
    """Instantiate dataloader in usable state."""
    dataset = make_dataset(tileshape)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=shuffle)
    return loader
    

def configure_benchmarking_environment(device: str | torch.device,
                                       torch_num_threads: int,
                                       torch_interop_threads: int) -> torch.device:
    device = torch.device(device) if not isinstance(device, torch.device) else device
    logger.info(f'Pre-configuration TORCH_NUM_THREADS = {torch.get_num_threads()}')
    logger.info(f'Pre-configuration TORCH_NUM_INTEROP_THREADS = {torch.get_num_interop_threads()}')
    logger.info(f'Benching with user-set TORCH_NUM_THREADS = {torch_num_threads}')
    logger.info(f'Benching with user-set TORCH_NUM_INTEROP_THREADS = {torch_interop_threads}')
    logger.info(f'Benching with user-set model pass device: {device}')
    torch.set_num_threads(torch_num_threads)
    torch.set_num_interop_threads(torch_interop_threads)
    return device



def load(model: torch.nn.Module,
         loader: torch.utils.data.DataLoader,
         parametrizations: Sequence[ParametrizedTransform],
         device: torch.device,
         dtype: torch.dtype,
         display_parametrizations_progress: bool = False,
         display_loader_progress: bool = True
         ) -> None:
    """
    Mock function with identical signature to `evaluate` to isolate
    data loading for benchmarking.
    This function more purely measures data yielding:
        -> retrieval from dataset
        -> transformation via static transforms and the parametrized transform
        -> transfer to device and cast to preset data type
    
    Many empty arguments are used for compatibility reasons.
    """
    assert len(parametrizations) == 1, 'benchmark expects single parametrized transform'
    parametrization = parametrizations[0]
    set_parametrized_transform(loader.dataset, transform=parametrization)

    if display_loader_progress:
        settings = {
        'desc' : 'loader progress', 'unit' : 'bt',
        'leave' : False
        }
        loader = tqdm.tqdm(loader, **settings)
    
    # use blocking transfer to measure dtype casts and transfer
    for batch in loader:
        data, label = batch
        data = data.to(device=device, dtype=dtype, non_blocking=False)
        label = label.to(device=device, dtype=dtype, non_blocking=False)

    return None



def run_benchmark(torch_num_threads: int,
                  torch_num_interop_threads: int,
                  tileshape: tuple[int, int, int],
                  batch_size: int,
                  num_workers: int,
                  device: str | torch.device,
                  size: int | None = None,
                  flavor: str = 'end2end',
                  dtype: torch.dtype = torch.float32) -> None:
    
    timestamp = create_timestamp()
    device = configure_benchmarking_environment(device,
                                                torch_num_threads=torch_num_threads,
                                                torch_interop_threads=torch_num_interop_threads)
    location = Path(__file__)
    result_dir = location / 'benchresults'
    result_dir.mkdir(exist_ok=True)

    length = len(loader.dataset)
    model = ResNet3D(in_channels=1)
    loader = make_loader(tileshape=tileshape, batch_size=batch_size,
                         num_workers=num_workers)
    parametrizations = make_smoother()

    if flavor == 'end2end':
        func = evaluate
    elif flavor == 'load_only':
        func = load
    else:
        raise ValueError(f'invalid benchmark flavor option: \'{flavor}\'')

    t_start = time.time()
    _ = func(model=model, loader=loader, parametrizations=parametrizations,
             device=device, dtype=dtype,
             display_parametrizations_progress=False,
             display_loader_progress=True)
    t_end = time.time()

    delta_t = t_end - t_start

    report ={
        'timestamp' : timestamp,
        'flavor' : flavor,
        'torch_num_threads' : torch_num_threads,
        'torch_num_interop_threads' : torch_num_interop_threads,
        'tileshape' : tileshape,
        'batch_size' : batch_size,
        'device' : device,
        'dset_length' : length,
        'preset_size' : size,
        'delta_t' : delta_t,
        'average_time_per_element' : delta_t / length 
    }

    rich.print(report)

    fname = f'bnmrk_{flavor}_{timestamp}.json'
    fpath = result_dir / fname
    with fpath.open(mode='w') as handle:
        json.dump(report, handle)



