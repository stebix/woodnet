import torch

from typing import Literal, Optional

from dataobjects import AbstractSlice




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
            if self.class