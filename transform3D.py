import random
import torch
from typing import Any, Iterable

Tensor = torch.Tensor


class Normalize3D:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


    def __call__(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std
    


class Rotate90:
    """
    Rotate by 90 degrees in the plane defined by the two given dims axis.
    TODO: give ability to seed the torch random number generator
    """
    def __init__(self, dims: Iterable[int]) -> None:
        self.dims = dims

    def __call__(self, x: Tensor) -> Tensor:
        k = random.randint(0, 3)
        if k == 0:
            return x
        
        # check if channel-wise rotation is required
        if x.ndim == 3:
            y = torch.rot90(x, k, dims=self.dims)
        elif x.ndim == 4:
            y = torch.stack([
                torch.rot90(x[c, ...], k, dims=self.dims)
                for c in range(x.shape[0])
            ])
        else:
            raise ValueError(f'expected 3D or 4D tensor but got shape {x.shape}')
        return y