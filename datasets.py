import torch
import numpy as np

from typing import Literal, Optional, Union

from dataobjects import AbstractSlice


DEFAULT_CLASSLABEL_MAPPING = {
    'ahorn' : 0,
    'kiefer' :  1
}

Tensor = torch.Tensor


class SliceDataset(torch.utils.data.Dataset):


    def __init__(self,
                 phase: Literal['train', 'test'],
                 slices: list[AbstractSlice],
                 classlabel_mapping: Optional[dict[str, int]] = None,
                 transformer: Optional[callable] = None,
                 ) -> None:
        
        self.phase = phase
        self.classlabel_mapping = classlabel_mapping
        self.transformer = transformer
        
        if self.phase == 'train':
            if self.classlabel_mapping is None:
                raise TypeError(
                    f'{self.__class__.__name__} requires class to label mapping '
                     'at for the selected train phase'
                )

        self.slices = slices


    def __getitem__(self, index: int) -> Union[Tensor, tuple[Tensor, int]]:
        """
        Return signature is phase-dependent:
            - train phase has tuple output
                 -> (input [Tensor, (H x W)], label [integer])
            - test phase has tensor output
                 ->  input [Tensor, (H x W)]
        """
        slc = self.slices[index]
        data = torch.tensor(slc.data)

        if self.transformer:
            data = self.transformer(data)

        if self.phase == 'test':
            return data

        label = self.classlabel_mapping[slc.class_]
        label = torch.tensor(label)

        return (data, label)
    

    def __len__(self) -> int:
        return len(self.slices)
        
        
        

