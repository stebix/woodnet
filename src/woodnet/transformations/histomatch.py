"""
Implement histogram matching for planar data.
"""
import abc
import logging
import time

from collections.abc import Sequence, Callable
from typing import Literal
from pathlib import Path

import numpy as np
import torch
import skimage.exposure
import monai
from torchvision.transforms import Compose

import woodnet.transformations.transforms


DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(__name__)



def get_gaussian_smooth_implementation(
        impl: Literal['monai', 'woodnet']
    ) -> type:
    """
    Get the Gaussian smooth implementation to use.
    """
    if impl == 'monai':
        return monai.transforms.GaussianSmooth
    elif impl == 'woodnet':
        return woodnet.transformations.transforms.GaussianBlur
    else:
        raise ValueError(
            f'Unknown Gaussian smooth implementation: {impl}. '
            f'Expected one of: \'monai\', \'woodnet\''
        )


def get_seed() -> int:
    """
    Get a random seed for reproducibility.
    """
    return int(time.time())


class _GetSeedMixin:
    """Enables logged seed generation for reproducibility."""
    def _get_seed(self) -> int:
        """
        Get a random seed for reproducibility.
        """
        seed = get_seed()
        logger.debug(
            f'Seed was not provided for {self.__class__.__name__}: using time-based seed: {seed}'
        )
        return seed



class AbstractReferenceProvider(abc.ABC):
    """
    Provide reference images for histogram matching.
    The reference image is can be a 2D (H x W) or 3D (H x W x C) image.
    
    Note that the class provides images in the directly *channels-last*
    format, i.e. (H x W x C). This is required for the
    `skimage.exposure.match_histograms` function!
    This is in contrast to the usual PyTorch format (C x [... spatial ...]). 
    """
    def __init__(
        self
    ) -> None:
        pass
    
    @abc.abstractmethod
    def retrieve(self, channels: Literal['single', 'all'] = 'all') -> np.ndarray:
        """
        Retrieve reference image for histogram matching.
        Can be 2D (H x W) or 3D (H x W x C), i.e. channels last.
        """
        raise NotImplementedError("ReferenceProvider must be implemented.")


class MockReferenceProvider(AbstractReferenceProvider):
    def __init__(self, image):
        self.image = image

    def retrieve(self, *args, **kwargs) -> np.ndarray:
        """
        Retrieve reference image for histogram matching.
        Can be 2D (H x W) or 3D (H x W x C), i.e. channels last.
        """
        return self.image


class ReferenceProvider(AbstractReferenceProvider):
    """
    Provide reference images for histogram matching.

    Input images held by this object are expected to be in the
    format (N x C x H x W).
    """
    def __init__(
        self,
        reference_images: np.ndarray,
        primary_channel: int = 0,
        seed: int | None = None
    ) -> None:
        try:
            ref_image_count, channel_count, _, _ = reference_images.shape
        except ValueError:
            raise ValueError(
                f'Reference images must be 4D and in the format (N x C x H x W) '
                f'but got {reference_images.ndim}D.'
            )
        if primary_channel >= channel_count or primary_channel < 0:
            raise ValueError(
                f'Primary channel {primary_channel} is out of bounds for {channel_count} channels.'
            )
        self.primary_channel = primary_channel
        self.reference_images = reference_images
        self._ref_image_count = ref_image_count
        self.seed = seed or get_seed()
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return self._ref_image_count

    def retrieve(self, channels: Literal['single', 'all'] = 'all') -> np.ndarray:
        """
        Retrieve a reference image for histogram matching.
        Depending on the channels argument, this can be a 2D (H x W) or
        3D (H x W x C) image, i.e. channels last.
        """
        index = self.rng.integers(0, self._ref_image_count)
        refimage = self.reference_images[index]
        if channels == 'single':
            # return only the primary channel
            refimage = refimage[self.primary_channel, ...]
        elif channels == 'all':
            refimage = np.moveaxis(refimage, 0, -1)
        else:
            raise ValueError(
                f'Unknown channel selection {channels}. '
                f'Expected one of: \'single\', \'all\''
            )
        return refimage

    def __str__(self) -> str:
        info_str = ''.join((self.__class__.__name__, '('))
        info_str += f'primary_channel={self.primary_channel}, '
        info_str += f'reference_image_count={self._ref_image_count}, '
        info_str += f'seed={self.seed}'
        return ''.join((info_str, ')'))
    
    def __repr__(self) -> str:
        return str(self)



class HistogramMatcher(_GetSeedMixin):
    """
    Apply histogram matching to a tensor using reference images from the
    ReferenceProvider.

    Note that the backend uses `skimage`, such that inputs are 
    converted to numpy arrays and back to torch tensors.
    If tensors are on GPU, tis introduces syncs and transfer costs! 
    """
    primary_axis: int = 0

    def __init__(
        self,
        p_execution: float,
        reference_provider: AbstractReferenceProvider,
        seed: int | None = None
    ) -> None:
        self.p_execution = p_execution
        self.reference_provider = reference_provider
        self.seed = seed or self._get_seed()
        self.rng = np.random.default_rng(self.seed)


    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Input expected to be:
            - (C x H x W) for triaxial data
            - (H x W) for planar data
        """
        if self.p_execution < self.rng.uniform(0, 1):
            return tensor
        
        if tensor.ndim == 3:
            # make triaxial data to channels last
            tensor = tensor.permute(1, 2, 0)
            channels = 'all'
        else:
            # planar data
            channels = 'single'
        
        array = tensor.cpu().numpy()
        reference = self.reference_provider.retrieve(channels)

        matched = skimage.exposure.match_histograms(array, reference)
        if matched.ndim == 3:
            # backtransform triaxial data to channels first
            matched = matched.transpose(2, 0, 1)
        # data typpe consistent to tensor input *should* be handled by skimage
        matched = torch.from_numpy(matched).to(tensor.device)
        # TODO: remove in production
        assert matched.dtype == tensor.dtype

        return matched
    
    def __str__(self) -> str:
        info_str = ''.join((self.__class__.__name__, '('))
        info_str += f'p_execution={self.p_execution}, '
        info_str += f'reference_provider={self.reference_provider}'
        return ''.join((info_str, ')'))
    
    def __repr__(self) -> str:
        return str(self)
        


class SystemEmulator(_GetSeedMixin):
    """
    Emulates a different CT system by applying resolution transformations
    like smoothing and intensity transformations like histogram matching
    and gamma adjustments.

    This is intended to emulate another CT system where both resolution
    and intensity characteristics are different due to measurement geometry,
    detector technology, and reconstruction algorithms.

    Histogram matching is done using `skimage` and uses numpy. If inputs are
    on GPU, this introduces syncs and transfer costs!

    Fake channel dimension is required for input data without channel axis/dim,
    since the monai Gaussian smooth implementation regards the first axis as
    the channel axis and does not use it for smoothing. When the fake channel
    dimension is not added, then we get anisotropic smoothing.
    """
    gaussian_smooth_implementation: Literal['monai'] = 'monai' 

    def __init__(
        self,
        p_execution: float,
        sigma: float | Sequence[float],
        reference_provider: AbstractReferenceProvider | None,
        gamma: float | None = None,
        add_fake_channel_dim: bool = True,
        seed: int | None = None
    ) -> None:
        
        self.p_execution = p_execution
        self.sigma = sigma if isinstance(sigma, Sequence) else [sigma]
        self.gamma = gamma
        self.add_fake_channel_dim = add_fake_channel_dim
        self.seed = seed or self._get_seed()
        self.rng = np.random.default_rng(self.seed)
        self.reference_provider = reference_provider or self._get_reference_provider()

        self.smoother = self._setup_gaussian_smooth()
        self.contrast_adjuster = self._setup_adjust_contrast()
        self.histogram_matcher = self._setup_histogram_matcher()


    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Tensor expected to be:
            - (C x H x W) for triaxial data
            - (H x W) for planar data
        """
        if tensor.ndim > 3:
            raise NotImplementedError(
                'currently only 3D and 4D tensors are supported: implement 3D histogram matching'
            )

        if self.p_execution < self.rng.uniform(0, 1):
            return tensor
        
        # hard-coded transformation order:
        # -> gaussian smoothing -> contrast adjustment -> histogram matching
        
        # TODO: Make this cool and without if statements
        # fake channel required for monai Gaussian smooth implementation,
        # since it regards the first axis as the channel axis
        do_fake_channel = tensor.ndim == 2 and self.add_fake_channel_dim
        if do_fake_channel:
            # add fake channel dimension
            tensor = tensor.unsqueeze(0)
        tensor = self.smoother(tensor)
        if do_fake_channel:
            # remove fake channel dimension
            tensor = tensor.squeeze(0)

        tensor = self.contrast_adjuster(tensor)
        tensor = self.histogram_matcher(tensor)
        return tensor
        
        
    def _setup_gaussian_smooth(self) -> Callable:
        """
        Setup Gaussian smoothing transform.
        """
        smoother_class = get_gaussian_smooth_implementation(
            self.gaussian_smooth_implementation
        )
        smoother = Compose([smoother_class(s) for s in self.sigma])
        return smoother
    
    def _setup_adjust_contrast(self) -> Callable:
        """
        Setup gamma adjustment transform.
        """
        if self.gamma is None:
            return lambda x: x
        else:
            return monai.transforms.AdjustContrast(gamma=self.gamma)
        
    def _setup_histogram_matcher(self) -> HistogramMatcher:
        """
        Setup histogram matching transform.
        """
        return HistogramMatcher(
            p_execution=self.p_execution,
            reference_provider=self.reference_provider,
            seed=self.seed
        )

    def _get_reference_provider(self) -> ReferenceProvider:
        # TODO: This is a dirty hack and we need to change this ASAP
        REFERENCE_IMAGES_PATH = Path(
            '/home/jannik/storage/wood-jnde/reference_images/refimages-v1.npy'
        )
        logger.info(f'using dirty hack to load ref image data from \'{REFERENCE_IMAGES_PATH}\'')
        reference_images = np.load(REFERENCE_IMAGES_PATH)
        reference_provider = ReferenceProvider(
            reference_images=reference_images,
            seed=self.seed
        )
        return reference_provider


    def __str__(self) -> str:
        info_str = ''.join((self.__class__.__name__, '('))
        info_str += f'p_execution={self.p_execution}, '
        info_str += f'smoothing={self._stringify_smoothers()}, '
        info_str += f'contrast_adjustment={self.gamma}, '
        info_str += f'histogram_matcher={self.histogram_matcher}, '
        info_str += f'add_fake_channel_dim={self.add_fake_channel_dim}'
        return ''.join((info_str, ')'))
    
    def __repr__(self) -> str:
        return str(self)

    def _stringify_smoothers(self) -> str:
        """Get a nice string representation of the smoothers"""
        if len(self.sigma) > 1:
            prefix = '['
            suffix = ']'
        else:
            prefix = ''
            suffix = ''

        smoothers_str = ', '.join(
            (f'GaussianSmooth(sigma={s})' for s in self.sigma)
        )
        return ''.join((prefix, smoothers_str, suffix))
