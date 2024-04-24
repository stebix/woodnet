"""
Top-level entry module for evaluation tasks.

@jsteb 2024
"""
import os
import logging
import json
import enum
import pickle

from collections.abc import Sequence, Mapping, Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from woodnet.inference.inference import run_evaluation, write_json, write_pickle
from woodnet.inference.evaluate import recursive_key_to_string, recursive_value_to_statedict
from woodnet.inference.directories import create_inference_directory
from woodnet.hooks import install_loginterceptor_excepthook
from woodnet.logtools.infrastructure import (create_logfile_name,
                                             create_logging_infrastructure,
                                             finalize_logging_infrastructure)
from woodnet.inference.parametrized_transforms import (CongruentParametrizations,
                                                       generate_parametrized_transforms)
from woodnet.globconf import (configure_torch_cpu_threading,
                              TORCH_NUM_THREADS, TORCH_NUM_INTEROP_THREADS)


FILE_LOCATION: Path = Path(__file__).parent
PRESETS_DIRECTORY: Path = FILE_LOCATION / 'inference/presets/'
if not PRESETS_DIRECTORY.is_dir():
    raise NotADirectoryError(
        f'Required presets directory not found @ \'{PRESETS_DIRECTORY}\''
    )


def load_parametrized_transforms_specs_from(preset_name: str) -> list[dict]:
    for item in PRESETS_DIRECTORY.iterdir():
        # use or with both conditions to allow user to select with
        # or without the file type suffix
        if preset_name == item.name or preset_name == item.stem:
            break
    else:
        raise FileNotFoundError(
            f'Could not localize preset with name \'{preset_name}\''
        )
    with item.open(mode='r') as handle:
        preset = json.load(handle)
    return preset



def generate_transforms_from(preset_name: str) -> Sequence[CongruentParametrizations]:
    """
    Automated and integrated generation of `transforms` from the preset name.
    """
    specs = load_parametrized_transforms_specs_from(preset_name)
    transforms = [
        CongruentParametrizations(transform)
        for transform in generate_parametrized_transforms(*specs, squeeze=False)
    ]
    return transforms


def init_device(device: torch.device | str, logger: logging.Logger) -> torch.device:
    if not isinstance(device, torch.device):
        device = torch.device(device)
    logger.info(f'Using device < {device} > for evaluation experiment')
    return device


def init_dtype(dtype: torch.dtype | str, logger: logging.Logger) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        'float32' : torch.float32,
        'float64' : torch.float64,
    }
    if not isinstance(dtype, torch.dtype):
        dtype = mapping[dtype]
    logger.info(f'Using data type < {dtype} > for evaluation experiment')
    return dtype


def all_isinstance(iterable: Iterable, class_or_tuple) -> bool:
    return all(isinstance(elem, class_or_tuple) for elem in iterable)

class StoreProtocol(enum.Enum):
    JSON = 'json'
    PICKLE = 'pickle'

suffixes: dict[StoreProtocol, str] = {
    StoreProtocol.JSON : 'json',
    StoreProtocol.PICKLE : 'pkl',
}

WriterFunc = Callable[[Any, Path], Path]

writers: dict[StoreProtocol, WriterFunc] = {
    StoreProtocol.JSON : write_json,
    StoreProtocol.PICKLE : write_pickle
}

def rename_write_failure(path: Path) -> Path:
    """
    Generates new filepath from the input by inserting informative string
    about the writer failure.
    """
    info = 'WRITEFAILURE-'
    return path.parent / f'{info}{path.name}'


def interpret_store_protocols(
        store_protocols: StoreProtocol | Sequence[StoreProtocol] | str | None
    ) -> list[StoreProtocol]:
    """
    Interpret heterogenous input(s) sensibly and mold them into single
    homogenous `list[StoreProtocol]` data layout.
    """
    # default: use all
    if store_protocols is None or store_protocols == 'all':
        store_protocols = list(StoreProtocol)
    # user-selected (sub)set of StoreProtocol members
    elif all_isinstance(store_protocols, StoreProtocol):
        store_protocols = list(store_protocols)
    # single string value representation
    elif isinstance(store_protocols, str):
        store_protocols = [StoreProtocol(store_protocols)]
    # single member of StoreProtocol
    elif isinstance(store_protocols, StoreProtocol):
        store_protocols = [store_protocols]
    else:
        raise TypeError(f'invalid store_protocols specification: \'{store_protocols}\'')
    return store_protocols


def write_results(data: Mapping,
                  protocols: Sequence[StoreProtocol],
                  directory: Path,
                  logger: logging.Logger) -> None:
    """Write the results to the directory."""
    filestem = 'evaluation'
    for protocol in protocols:
        writer = writers[protocol]
        suffix = suffixes[protocol]
        filepath = directory / '.'.join((filestem, suffix))
        try:
            writer(data, filepath)

        except (TypeError, ValueError) as e:
            logger.warning(f'Could not write data with {protocol} due to \'{e}\'. '
                           f'Attempting so salvage with data dict remodulation.')
            data_remod = recursive_key_to_string(recursive_value_to_statedict(data))
            # clean up any partially written files
            if filepath.is_file():
                src = filepath
                dst = rename_write_failure(filepath)
                os.rename(src=src, dst=dst)
                logger.info(f'Moved partially written file from \'{src}\' to \'{dst}\'')
                
            # deeply nested, but I cannot think of a more elegant solution right now
            try:
                writer(data_remod, filepath)
            except (pickle.PickleError, TypeError, ValueError) as e:
                    logger.error(
                        f'Failed to write data with {protocol} to '
                        f'\'{filepath}\' due to \'{e}\'')
            continue

        except pickle.PickleError as e:
            logger.error(
                f'Failed to write data with {protocol} to \'{filepath}\' due to \'{e}\''
            )
            continue
        logger.info(f'Successfully written data with {protocol} to \'{filepath}\'')


def run_evaluation_experiment(basedir: str | Path,
                              transforms_preset: str,
                              batch_size: int,
                              device: torch.device | str,
                              dtype: torch.dtype | str,
                              use_amp: bool,
                              use_inference_mode: bool,
                              non_blocking_transfer: bool = True,
                              num_workers: int = 0,
                              shuffle: int = False,
                              pin_memory: bool =False,
                              no_compile_override: bool = False,
                              global_torchconf: Mapping | None = None,
                              store_protocols: StoreProtocol | Sequence[StoreProtocol] | str | None = None,
                              inject_early_to_device: bool = False
                              ) -> None:
    """
    Performs a fully automated evaluation experiment run with/in the `basedir`.

    Different fold-wise subexperiments and model instances are evaluated automatically
    with the parametrized transforms loaded from the indicated preset.
    """
    basedir = Path(basedir)
    if not basedir.is_dir():
        raise NotADirectoryError(f'Invalid tentative base directory \'{basedir}\'')
    
    store_protocols = interpret_store_protocols(store_protocols)
    levels = {
        'DEBUG' : logging.DEBUG,
        'INFO' : logging.INFO,
        'WARNING' : logging.WARNING,
        'ERROR' : logging.ERROR,
        'CRITICAL' : logging.CRITICAL
    }
    stream_log_level = levels.get(os.environ.get('STREAM_LOG_LEVEL', None), logging.ERROR)
    level = logging.DEBUG
    (logger,
     streamhandler,
     memoryhandler) = create_logging_infrastructure(level=level,
                                                    streamhandler_level=stream_log_level)
    install_loginterceptor_excepthook(logger)

    # configure CPU thread parallelism before any significant torch action 
    # -> otherwise will have no effect
    global_torchconf = global_torchconf or {}
    num_threads = global_torchconf.get('torch_num_threads', None) or TORCH_NUM_THREADS
    num_interop_threads = global_torchconf.get('torch_num_interop_threads' or TORCH_NUM_INTEROP_THREADS)
    configure_torch_cpu_threading(num_threads, num_interop_threads)

    transforms = generate_transforms_from(transforms_preset)

    # finalize logging infrastructure -> start writuing to logfile on disk
    inference_directory = create_inference_directory(basedir)
    logfile_path = inference_directory / create_logfile_name()
    finalize_logging_infrastructure(logger, memoryhandler, logfile_path)

    # configure and log a whole lot of stuff
    device = init_device(device, logger)
    dtype = init_dtype(dtype, logger)
    logger.info(f'Using batch size b = {batch_size}')
    logger.info(f'Using automatic mixed precision setting use_amp = {use_amp}')
    if use_inference_mode:
        logger.info(f'Using optimized inference mode')
    else:
        logger.info(f'Using standard no_grad mode')
    logger.info(f'Using N = {num_workers} data loader worker processes')
    logger.info(f'Using setting \'{non_blocking_transfer=}\'')
    logger.info(f'Using setting \'{shuffle=}\'')
    logger.info(f'Using setting \'{pin_memory=}\'')
    logger.info(f'Using setting \'{no_compile_override=}\'')
    logger.info(f'Results will be stored with protocols: \'{store_protocols}\'')

    # actual core computation
    result = run_evaluation(basedir=basedir,
                            transforms=transforms,
                            batch_size=batch_size,
                            device=device,
                            dtype=dtype,
                            use_amp=use_amp,
                            use_inference_mode=use_inference_mode,
                            non_blocking_transfer=non_blocking_transfer,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory,
                            no_compile_override=no_compile_override,
                            inject_early_to_device=inject_early_to_device)

    logger.info(f'Concluded evaluation run for basedir \'{basedir}\'')
    write_results(result,
                  protocols=store_protocols,
                  directory=inference_directory, logger=logger)
    logger.info(f'Successful finalized evaluation experiment')

