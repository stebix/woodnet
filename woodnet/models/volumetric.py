import torch

from typing import Type, Literal

from woodnet.models.buildingblocks import ResNetBlock, create_activation

Tensor = torch.Tensor


MODULE_LOGGER_NAME: str = '.'.join(('main', __name__))


class ResNet3D(torch.torch.nn.Module):
    """
    Smallish volumetric ResNet model.
    """
    # ResNet 18 settings, see paper
    num_classes: int = 1
    num_layers: int = 18
    layers = [2, 2, 2, 2]
    expansion: int = 1
    conv_1_channels: int = 64
    channel_cascade = [64, 128, 256, 512]
    _dimensionality: Literal['3D'] = '3D'

    conv_class: torch.nn.Module = torch.nn.Conv3d
    norm_class: torch.nn.Module = torch.nn.BatchNorm3d

    def __init__(self, 
                 in_channels: int,
                 block: Type[ResNetBlock] = ResNetBlock,
                 final_nonlinearity: str = 'sigmoid',
                 final_nonlinearty_kwargs: dict | None = None,
                 testing: bool = False) -> None:

        super(ResNet3D, self).__init__()

        self.logger = '.'.join((MODULE_LOGGER_NAME, self.__class__.__name__))
        self.testing = testing

        self.global_in_channels = in_channels
        self.in_channels = self.conv_1_channels

        # Large initial convolution
        self.conv_1 = self.conv_class(in_channels=self.global_in_channels,
                                      out_channels=self.conv_1_channels,
                                      kernel_size=7, 
                                      stride=2,
                                      padding=3,
                                      bias=False
        )
        self.bn_1 = self.norm_class(self.conv_1_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer_1 = self._make_layer(block, 64, self.layers[0])
        self.layer_2 = self._make_layer(block, 128, self.layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, self.layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, self.layers[3], stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = torch.nn.Linear(512*self.expansion, self.num_classes)

        kwargs = final_nonlinearty_kwargs or {}
        self.final_nonlinearity = create_activation(final_nonlinearity, **kwargs)


    def _make_layer(self, block: Type[ResNetBlock], out_channels: int,
                    blocks: int, stride: int = 1) -> torch.nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = torch.nn.Sequential(
                self.conv_class(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                self.norm_class(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample,
                dimensionality=self._dimensionality
            )
        )
        self.in_channels = out_channels * self.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion,
                dimensionality=self._dimensionality
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

        if self.testing and self.final_nonlinearity is not None:
            x = self.final_nonlinearity(x)

        return x


    def forward_shapedebug(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        print(f'Out :: {x.shape}')

        x = self.bn_1(x)
        x = self.relu(x)

        print(f'In :: {x.shape}')
        x = self.maxpool(x)
        print(f'Out :: {x.shape}')
        x = self.layer_1(x)
        print(f'Out :: {x.shape}')
        x = self.layer_2(x)
        print(f'Out :: {x.shape}')
        x = self.layer_3(x)
        print(f'Out :: {x.shape}')
        x = self.layer_4(x)
        print(f'Out :: {x.shape}')
        x = self.avgpool(x)
        print(f'Out :: {x.shape}')
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
