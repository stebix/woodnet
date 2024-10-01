"""
Implement transformations for 3D volume data.

Jannik Stebani 2023
"""
import logging
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from collections.abc import Iterable

Tensor = torch.Tensor

DEFAULT_LOGGER_NAME = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

DEFAULT_SEED = int(time.time())
logger.info(f'Using time based RNG seed {DEFAULT_SEED}')


class Identity:
    def __init__(self):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return x
    
    def __str__(self) -> str:
        return f'Identity()'


class ToDevice:
    """Eagerly move incoming tensors to the indicated device."""
    def __init__(self,
                 device: torch.device,
                 non_blocking: bool = True) -> None:
        self.device = device
        self.non_blocking = non_blocking
    
    def __call__(self, x: Tensor) -> Tensor:
        x = x.to(device=self.device, non_blocking=self.non_blocking)
        return x
    
    def __str__(self) -> str:
        return f'ToDevice(device=\'{self.device}\', non_blocking={self.non_blocking})'
    
    def __repr__(self) -> str:
        return str(self)



class Normalize:
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



def Normalize3D(mean: float, std: float):
    logger.warning('Usage of class \'Normalize3D\' is deprecated. Use '
                   '\'Normalize\' instead.')
    return Normalize(mean=mean, std=std)



class Rotate90:
    """
    Rotate by 90 degrees in the plane defined by the two given dims axis.
    TODO: give ability to seed the torch random number generator
    """
    def __init__(self, dims: Iterable[int], seed: int = 123) -> None:
        self.dims = dims
        self.seed = seed

    def __call__(self, x: Tensor) -> Tensor:
        k = random.randint(0, 3)
        if k == 0:
            return x
        
        # check if channel-wise rotation is required
        if x.ndim == 3:
            y = torch.rot90(x, k, dims=self.dims)
        elif x.ndim == 4:
            y = torch.stack(
                [
                    torch.rot90(x[c, ...], k, dims=self.dims)
                    for c in range(x.shape[0])
                ]
            )
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



class GaussianNoise:
    """Additive Gaussian noise."""

    def __init__(self, p_execution: float, mean: float, std: float,
                 seed: int = 123) -> None:
        
        self.p_execution = p_execution
        self.mean = mean
        self.std = std
        self._seed = seed

    def __call__(self, x: Tensor) -> Tensor:
        if self.p_execution < random.uniform(0, 1):
            return x
        
        template = torch.ones_like(x)
        mean = self.mean * template
        std = self.std * template
        noise = torch.normal(mean=mean, std=std)
        return x + noise


    def __str__(self) -> str:
        s = (f'{self.__class__.__name__}(p_execution={self.p_execution:.3f}, '
             f'mean={self.mean}, std={self.std})')
        return s
    

    def __repr__(self) -> str:
        return str(self)



class PoissonNoise:
    """Additive Poisson noise."""

    def __init__(self, p_execution: float, lambda_: float,
                 seed: int = 123) -> None:
        
        self.p_execution = p_execution
        self.lambda_ = lambda_
        self._seed = seed

    def __call__(self, x: Tensor) -> Tensor:
        if self.p_execution < random.uniform(0, 1):
            return x
        
        template = self.lambda_ * torch.ones_like(x)
        noise = torch.poisson(template)
        return x + noise


    def __str__(self) -> str:
        s = (f'{self.__class__.__name__}(p_execution={self.p_execution:.3f}, '
             f'lambda={self.lambda_})')
        return s
    

    def __repr__(self) -> str:
        return str(self)


def convolve(x: Tensor, kernel: Tensor, padding: int) -> Tensor:
    """Compute 3D convolution as series of 1D convolution: separation theorem."""
    for _ in range(3):
        x = F.conv1d(x.reshape(-1, 1, x.size(2)), weight=kernel, padding=padding).view(*x.shape)
        x = x.permute(2, 0, 1)
    return x


class GaussianBlur:
    """
    Stochastically apply Gaussian blur to 4D or 5D ([N x] C x D x H x W) tensors.
    """
    def __init__(self, p_execution: float, ksize: int, sigma: int,
                 dtype: torch.dtype = torch.float32) -> None:

        self.p_execution = p_execution

        if ksize % 2 == 0:
            raise ValueError(f'kernel size should be odd, but got {ksize = }')
        self._sigma = sigma
        self._ksize = ksize
        self._dtype = dtype
    
        self._x = np.linspace(-ksize//2, +ksize//2, num=ksize)
        kernel = np.exp(-self._x**2 / sigma)
        kernel = kernel / np.sum(kernel)
        self._kernel = torch.tensor(
            kernel[np.newaxis, np.newaxis, ...], dtype=self._dtype
        )
        

    def __call__(self, x: Tensor) -> Tensor:
        """Execute transformation on tensor."""
        if x.ndim not in {3, 4}:
            raise ValueError(f'tensor must be 3D or 4D of ([C x] D x H x W) layout, but '
                             f'got ndim = {x.ndim}')
            
        if self.p_execution < random.uniform(0, 1):
            return x
        
        kernel = self._kernel.to(device=x.device, dtype=x.dtype)
        
        if x.ndim == 3:
            # perform 3D convolution as series of 1D convoltuions -> separability theorem
            for _ in range(3):
                x = convolve(x, kernel, padding=self._ksize//2)
        else:
            channels = []
            for c in range(x.size(0)):
                v = x[c, ...]
                channels.append(convolve(v, kernel, padding=self._ksize//2))
                
            x = torch.stack(channels, dim=0)
        return x



class LowResolution:
    """
    Simulate lower resolution input data by downsampling data and
    then upsampling again, leading to aliasing and image detail loss.
    """
    def __init__(self,
                 scale_factor: float,
                 downsample_interp_mode: str = 'nearest',
                 upsample_interp_mode: str = 'bilinear',
                 align_corners: bool = False,
                 antialias: bool = False,
                 use_fake_batch_dim: bool = False
                ) -> None: 
        self.scale_factor = scale_factor
        self.downsample_interp_mode = downsample_interp_mode
        self.upsample_interp_mode = upsample_interp_mode
        self.align_corners = align_corners
        self.antialias = antialias
        self.use_fake_batch_dim = use_fake_batch_dim
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # raw input from dataset has expected layout ([N x] C x D [x H x W])
        if self.use_fake_batch_dim:
            x = x.unsqueeze(dim=0)
        # save spatial shape of input, expected layout (N x C x D [x H x W])
        input_shape = x.shape[2:]
        lowered_shape = tuple(
            int(elem)
            for elem in np.round(np.array(input_shape) * self.scale_factor)
        )        
        downsampled = torch.nn.functional.interpolate(x, size=lowered_shape,
                                                      mode=self.downsample_interp_mode,
                                                      antialias=self.antialias)
        upsampled = torch.nn.functional.interpolate(downsampled, size=input_shape,
                                                    mode=self.upsample_interp_mode,
                                                    antialias=self.antialias,
                                                    align_corners=self.align_corners)
        if upsampled.shape != x.shape:
            logger.warning(
                f'Failed to regain original shape {x.shape} after downsampling to '
                f'{downsampled.shape}. Actual output shape was: {upsampled.shape}.'
            )
        if self.use_fake_batch_dim:
            upsampled = torch.squeeze(upsampled, dim=0)
        return upsampled