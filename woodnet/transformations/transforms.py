"""
Implement transformations for 3D volume data.

Jannik Stebani 2023
"""
import random
import torch

from collections.abc import Iterable

Tensor = torch.Tensor


class Normalize3D:
    """
    Transform (volume) histogram to zero mean and unit standard
    deviation parameters.
    """
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std
    
    def __str__(self) -> str:
        s = (f'{self.__class__.__name__}(mean={self.mean:.3g}, '
             f'std={self.std:.3g})')
        return s
    
    def __repr__(self) -> str:
        return str(self)


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
    

    def __str__(self) -> str:
        s = (f'{self.__class__.__name__}(dim={tuple(self.dims)})')
        return s
    

    def __repr__(self) -> str:
        return str(self)
    


class Rotate:
    def __init__(self) -> None:
        raise NotImplementedError('implement this')


class GaussianBlur:
    def __init__(self) -> None:
        raise NotImplementedError('implement this')


class GaussianNoise:
    def __init__(self) -> None:
        raise NotImplementedError('implement this')


class PoissonNoise:
    def __init__(self) -> None:
        raise NotImplementedError('implement this')

