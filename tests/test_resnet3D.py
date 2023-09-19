import torch


from woodnet.models.volumetric import ResNet3D


def test_resnet3D_smoke_init():
    s = 256
    data = torch.randn(size=(1, 1, s, s, s))
    model = ResNet3D(in_channels=1)

    print(model)

    with torch.no_grad():
        out = model(data)
        
    print(f'out shape = {out.shape}')