import torch


from models.volume import ResNet3D

s = 256

data = torch.randn(size=(1, 1, s, s, s))
model = ResNet3D(in_channels=1)

print(model)

with torch.no_grad():
    out = model(data)


print(f'out shape = {out.shape}')