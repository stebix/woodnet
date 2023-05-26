import datetime
import torch

from pathlib import Path

from custom.types import PathLike

class DirectoryHandler:
    allow_preexisting_dir: bool = False
    allow_overwrite: bool = False

    def __init__(self,
                 directory: PathLike) -> None:
        
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=self.allow_preexisting_dir)


    def initialize_subdirectories(self) -> None:
        log_dir = 'logs'
        checkpoint_dir = 'checkpoints'

        self.logs = self.directory / log_dir
        self.logs.mkdir()

        self.checkpoints = self.directory / checkpoint_dir
        self.checkpoints.mkdir()

    
    def save_model_checkpoint(self,
                              model: torch.nn.Module,
                              filename: str) -> None:
        savepath = self.checkpoints / filename
        if savepath.exists() and not self.allow_overwrite:
            raise FileExistsError(
                f'Could not save model checkpoint to "{savepath}"!'
                 'File already exists.'
            )
        torch.save(model.state_dict(), savepath)