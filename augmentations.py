from typing import Any
import torch



class Transformer:

    def __init__(self, *transforms: torch.nn.Module) -> None:
        self.transforms = transforms

    def __call__(self, data: torch.Tensor) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def __str__(self) -> str:
        info_str = ''.join((self.__class__.__name__, '('))
        info_str += f'N={len(self.transforms)}'
        





class ScriptedTransformer:

    def __init__(self, *transforms: torch.nn.Module) -> None:
        self.transforms = torch.nn.Sequential(*transforms)
        pass