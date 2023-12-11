import dataclasses
import datetime
import torch

from pathlib import Path

from custom.types import PathLike


@dataclasses.dataclass
class Directories:
    base: Path
    checkpoint: Path
    log: Path


class IOHandler:
    allow_preexisting_dir: bool = False
    allow_overwrite: bool = False

    logdir_name: str = 'logs'
    checkpointdir_name: str = 'checkpoints'

    def __init__(self,
                 directory: PathLike) -> None:
        
        self.dirs = self.initialize_directories(directory)


    def initialize_directories(self, base) -> None:
        base = Path(base)
        base.mkdir(exist_ok=self.allow_preexisting_dir)

        log = base / self.logdir_name
        log.mkdir()

        checkpoint = base / self.checkpointdir_name
        checkpoint.mkdir()

        return Directories(base=base, checkpoint=checkpoint, log=log)


    
    def save_model_checkpoint(self, model: torch.nn.Module, name: str) -> None:
        savepath = self.dirs.checkpoint / name
        if savepath.exists():
            raise FileExistsError(f'cannot save model at "{savepath}"')
        torch.save(model.state_dict(), savepath)