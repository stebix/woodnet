from typing import Any
import torch

Tensor = torch.Tensor



class Normalize3D:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


    def __call__(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std