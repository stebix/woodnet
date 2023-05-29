import torch
from typing import Optional, Literal


Tensor = torch.Tensor


def fetch_convolution_class(dimensionality: str) -> torch.nn.Module:
    if dimensionality == '2D':
        return torch.nn.Conv2d
    elif dimensionality == '3D':
        return torch.nn.Conv3d
    else:
        raise ValueError(f'invalid dimensionality: "{dimensionality}"')


def fetch_batchnorm_class(dimensionality: str) -> torch.nn.Module:
    if dimensionality == '2D':
        return torch.nn.BatchNorm2d
    elif dimensionality == '3D':
        return torch.nn.BatchNorm3d
    else:
        raise ValueError(f'invalid dimensionality: "{dimensionality}"')



class ResNetBlock(torch.torch.nn.Module):
    """
    Basic ResNet block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 expansion: int = 1,
                 downsample: Optional[torch.torch.nn.Module] = None,
                 dimensionality: Literal['2D', '3D'] = '2D',
                 ) -> None:
        super(ResNetBlock, self).__init__()

        conv_class = fetch_convolution_class(dimensionality)
        norm_class = fetch_batchnorm_class(dimensionality)


        # Multiplicative factor for the subsequent
        # intra-block expansion of the convolution layer channels
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv_1 = conv_class(in_channels, out_channels, 
                                 kernel_size=3, 
                                 stride=stride, 
                                 padding=1,
                                 bias=False
        )
        self.bn_1 = norm_class(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv_2 = conv_class(out_channels, out_channels*self.expansion, 
                                 kernel_size=3, 
                                 padding=1,
                                 bias=False
        )
        self.bn_2 = norm_class(out_channels*self.expansion)


    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # progress through the varying operations
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out
    
