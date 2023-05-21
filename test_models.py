import torch

from models import ResNet18

def test_ResNet18():
    """BAsic smoke test if forward pass goes through."""
    in_channels = 1
    out_channels = 2
    batch_size = 3
    size = (batch_size, in_channels, 256, 256)
    inputs = torch.randn(size=size)

    model = ResNet18(in_channels=in_channels)

    with torch.no_grad():
        prediction = model(inputs)
    
    assert prediction.shape[0] == batch_size
    assert prediction.shape[1] == out_channels