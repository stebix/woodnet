import logging
import torch

from typing import Type, Optional

from .buildingblocks import ResNetBlock, create_activation

Tensor = torch.Tensor

MODULE_LOGGER_NAME: str = '.'.join(('main', __name__))


class ResNet18(torch.torch.nn.Module):
    """
    Smallish 2D ResNet18 model.
    """
    # ResNet 18 settings, see paper
    num_classes: int = 1
    num_layers: int = 18
    layers = [2, 2, 2, 2]
    expansion: int = 1
    conv_1_channels: int = 64
    channel_cascade = [64, 128, 256, 512]
    _dimensionality: str = '2D'

    def __init__(self, 
                 in_channels: int,
                 block: Type[ResNetBlock] = ResNetBlock,
                 final_nonlinearity: str = 'sigmoid',
                 final_nonlinearity_kwargs: dict | None = None,
                 testing: bool = False) -> None:
        super(ResNet18, self).__init__()

        self.logger = '.'.join((MODULE_LOGGER_NAME, self.__class__.__name__))
        self.testing = testing

        self.global_in_channels = in_channels
        self.in_channels = self.conv_1_channels
        
        # Large initial convolution
        self.conv_1 = torch.nn.Conv2d(in_channels=self.global_in_channels,
                                      out_channels=self.conv_1_channels,
                                      kernel_size=7, 
                                      stride=2,
                                      padding=3,
                                      bias=False
        )
        self.bn_1 = torch.nn.BatchNorm2d(self.conv_1_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_1 = self._make_layer(block, 64, self.layers[0])
        self.layer_2 = self._make_layer(block, 128, self.layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, self.layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, self.layers[3], stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512*self.expansion, self.num_classes)
        
        kwargs = final_nonlinearity_kwargs or {}
        self.final_nonlinearty = create_activation(final_nonlinearity, **kwargs)


    def _make_layer(self, block: Type[ResNetBlock], out_channels: int,
                    blocks: int, stride: int = 1) -> torch.nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                torch.nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return torch.nn.Sequential(*layers)
    

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        if self.testing and self.final_nonlinearty is not None:
            x = self.final_nonlinearty(x)

        return x
