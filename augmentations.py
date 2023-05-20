from typing import Any
import torch



class Transformer:

    def __init__(self, *transforms: torch.nn.Module) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x
    
    def __str__(self) -> str:
        info_str = ''.join((self.__class__.__name__, '('))
        info_str += f'N={len(self.transforms)}'
        





class ScriptedTransformer:

    def __init__(self, *transforms: torch.nn.Module) -> None:
        self.transforms = torch.nn.Sequential(*transforms)
        self.scripted_transforms = torch.jit.script(self.transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scripted_transforms(x)