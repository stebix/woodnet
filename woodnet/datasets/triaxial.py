"""
Implements the triaxial dataset that utilizes the three orthogonal X/Y/Z-planes
inside the specified volume. 
"""
import numpy as np
import torch
import torch.utils.data as torchdata

from pathlib import Path
from typing import Literal, Callable

from woodnet.custom.types import PathLike


class TriaxialDataset(torchdata.Dataset):
    """
    Triaxial dataset with orthogonal images concatenated along the
    channel dimension.

    The dataset can provide the orthogonal images 
    
    """
    def __init__(self,
                 path: PathLike,
                 phase: Literal['train', 'val'],
                 tileshape: tuple[int] | None = None) -> None:

        super().__init__()

        self.path = Path(path)