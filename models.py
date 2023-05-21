import torch

from typing import Type, Optional



Tensor = torch.Tensor




class ResNetBlock(torch.torch.nn.Module):
    """
    Basic ResNet block.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 expansion: int = 1,
                 downsample: Optional[torch.torch.nn.Module] = None
                 ) -> None:
        super(ResNetBlock, self).__init__()
        # Multiplicative factor for the subsequent
        # intra-block expansion of the convolution layer channels
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv_1 = torch.torch.nn.Conv2d(in_channels, out_channels, 
                                            kernel_size=3, 
                                            stride=stride, 
                                            padding=1,
                                            bias=False
        )
        self.bn_1 = torch.torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.torch.nn.ReLU(inplace=True)
        self.conv_2 = torch.torch.nn.Conv2d(out_channels, out_channels*self.expansion, 
                                            kernel_size=3, 
                                            padding=1,
                                            bias=False
        )
        self.bn_2 = torch.torch.nn.BatchNorm2d(out_channels*self.expansion)


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
    



class ResNet18(torch.torch.nn.Module):
    """
    Smallish ResNet18 model.
    """
    # ResNet 18 settings, see paper
    num_classes: int = 1
    num_layers: int = 18
    layers = [2, 2, 2, 2]
    expansion: int = 1
    conv_1_channels: int = 64
    channel_cascade = [64, 128, 256, 512]

    def __init__(self, 
                 in_channels: int,
                 block: Type[ResNetBlock] = ResNetBlock) -> None:
        super(ResNet18, self).__init__()
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
        return x
