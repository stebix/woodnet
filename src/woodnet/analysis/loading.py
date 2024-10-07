"""
Tooling to programmatically load and process evaluation run results from the folder structure on the file
system.

@jsteb 2024
"""
import dataclasses
import logging
import pickle
import pandas as pd
import json

from collections.abc import Sequence, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from woodnet.utils import DEFAULT_TIMESTAMP_FORMAT
from woodnet.analysis.dataframetools import load_from_pkl

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def require_unique_and_existing(paths: Sequence[Path]) -> Path:
    if len(paths) == 1:
        path = paths[0]
        if not path.is_file():
          raise FileNotFoundError(f'specified path \'{path}\' does not exist on the file system')  
        return path
    if len(paths) == 0:
        raise FileNotFoundError(f'empty path specification')  
    raise ValueError(f'received non-unique path specification: \'{paths}\'')


@dataclasses.dataclass
class EvaluationBag:
    preset: str | None
    pickle_fpath: Path
    json_fpath: Path
    log_fpath: Path
    timestamp: datetime
        
    def fetch(self, item: str = 'pickle') -> dict | list[str]:
        """
        Fetch on of the three important elemtns of the evalutation bag.
        Either the pickled results file, the JSON results file or the log file.
        """
        if item == 'pickle':
            return self.fetch_pickle()
        elif item == 'json':
            return self.fetch_json()
        elif item == 'log':
            return self.fetch_log()
        else:
            raise ValueError(f'invalid item specification: \'{item}\'')
    
    def fetch_pickle(self) -> dict:
        with self.pickle_fpath.open(mode='rb') as handle:
            return pickle.load(handle)
    
    def fetch_json(self) -> dict:
        with self.json_fpath.open(mode='r') as handle:
            return json.load(handle)
    
    def fetch_log(self) -> list[str]:
        with self.log_fpath.open(mode='r') as handle:
            return handle.readlines()
    
    @property
    def timestamp_str(self) -> str:
        return self.timestamp.strftime(DEFAULT_TIMESTAMP_FORMAT)
    
    @classmethod
    def from_directory(cls, directory: Path, timestamp: datetime) -> 'EvaluationBag':
        try:
            preset_artifact = require_unique_and_existing(list(directory.glob('PRESETARTIFACT_*')))
        except FileNotFoundError as e:
            logger.warning(
                f'could not retrieve preset artifact for directory: \'{directory}\' '
                f'due to error reason: \'{e}\''
            )
            preset = None
        else:
            _, preset = preset_artifact.name.split('_')

        log_fpath = require_unique_and_existing(list(directory.glob('*.log')))
        pickle_fpath = require_unique_and_existing(list(directory.glob('*.pkl')))
        json_fpath = require_unique_and_existing([directory / 'evaluation.json'])
        return cls(preset=preset, pickle_fpath=pickle_fpath, json_fpath=json_fpath,
                   log_fpath=log_fpath, timestamp=timestamp)
        




def load_inference_results(basedir: Path | str) -> dict[datetime, Path]:
    """Load the inference results from the basal experiment directory."""
    runs: dict[datetime.datetime, Path] = {}
    basedir = Path(basedir) if not isinstance(basedir, Path) else basedir
    inference_dir = basedir / 'inference'
    if not inference_dir.is_dir():
        raise NotADirectoryError(f'Provided base directory \'{basedir}\' does not contain a inference directory')
        
    for rundir in inference_dir.iterdir():
        if not rundir.is_dir():
            logger.warning(
                f'Ignoring item \'{rundir}\' in basal dir \'{basedir}\'. Reason: is not a directory.'
            )
            continue
        try:
            timestamp = datetime.strptime(rundir.name, DEFAULT_TIMESTAMP_FORMAT)
        except ValueError as e:
            logger.warning(
                f'Ignoring item \'{rundir}\' in basal dir \'{basedir}\'. Reason: unparseable '
                f'timestamp string via exception \'{e}\''
            )
        runs[timestamp] = rundir
    return runs


def retrieve_preset_name(rundir: Path) -> str | None:
    preset_name = None
    for item in rundir.iterdir():
        if item.name.startswith('PRESETARTIFACT'):
            _, preset_name = item.name.split('_')
            break
    return preset_name
    


def filter_for_preset(mapping: Mapping[Any, Path], preset: str = 'any') -> Mapping[Any,Path]:
    """
    Select only the inference result directories that contain the preset artifact with
    matching name inside them.
    """
    if preset == 'any':
        return mapping
    filtered = {}
    for key, dirpath in mapping.items():

        current_preset = retrieve_preset_name(dirpath)
        if current_preset is None or current_preset != preset:
            continue

        filtered[key] = dirpath

    if not filtered:
        raise FileNotFoundError(f'no matching inference result found for preset \'{preset}\'')

    return filtered

        
def retrieve_newest(inference_results: Mapping[datetime, Path]) -> tuple[datetime, Path]:
    """Retrieve the newest/latest inference run timestamp-directory pair from the mapping."""
    recent = max(timestamp for timestamp in inference_results.keys())
    return (recent, inference_results[recent])


def load_newest(basedir: Path | str, preset: str = 'any') -> EvaluationBag:
    """Load the newest evaluation bag (thin information struct) from the base directory."""
    runs = load_inference_results(basedir)
    runs = filter_for_preset(runs, preset=preset)
    timestamp, path = retrieve_newest(runs)
    evalbag = EvaluationBag.from_directory(directory=path, timestamp=timestamp)
    return evalbag


def load_newest_evaluation_df(basedir: Path | str, preset: str = 'any') -> tuple[pd.DataFrame, EvaluationBag]:
    """
    Loads and preprocesses the results from the newest evaluation run as a readily usable
    pandas dataframe. Also, the metadata-containing evaluation bag is returned.
    """
    evalbag = load_newest(basedir, preset=preset)
    df = load_from_pkl(filepath=evalbag.pickle_fpath)
    return (df, evalbag)

    