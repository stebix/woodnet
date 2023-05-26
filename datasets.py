import torch
import numpy as np

from typing import Literal, Optional, Union

from dataobjects import AbstractSlice, Volume


DEFAULT_CLASSLABEL_MAPPING = {
    'ahorn' : 0,
    'kiefer' :  1
}

Tensor = torch.Tensor


def add_channel_dim(array: np.ndarray) -> np.ndarray:
    """Add fake channel dimension."""
    return array[np.newaxis, ...]


class SliceDataset(torch.utils.data.Dataset):
    """
    Dataset consisting of many 2D slices (H x W)
    """
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
        data = torch.tensor(add_channel_dim(slc.data))

        if self.transformer:
            data = self.transformer(data)

        if self.phase == 'test':
            return data

        label = torch.tensor(self.classlabel_mapping[slc.class_]).unsqueeze(-1)

        return (data, label.to(torch.long))
    

    def __len__(self) -> int:
        return len(self.slices)
        
        
        

class EagerSliceDataset(torch.utils.data.Dataset):
    """
    Eager version of the slice dataset that loads the full volume into main
    memory as numpy.ndarray.

    Useful for repeated prediction e.g. in the context of a 
    jupyter notebook environment.
    """
    # TODO: Refactor constructor without expicit dependency on volume
    def __init__(self,
                 phase: Literal['train', 'val'],
                 volume: Volume,
                 transformer: Optional[callable] = None,
                 classlabel_mapping: Optional[dict[str, int]] = None,
                 axis: int = 0,
                 ) -> None:

        self.phase = phase
        self.axis = axis
        self.fingerpint = volume.fingerprint
        self.volume = np.swapaxes(volume.data, 0, axis)
        self.transformer = transformer

        if self.phase == 'train' and classlabel_mapping is None:
            raise RuntimeError('Training phase dataset requires a '
                               'classlabel mapping!')

        self.classlabel_mapping = classlabel_mapping
        # we assume that for the loaded volume the class is
        #  constant 
        self._label = torch.tensor(
            self.classlabel_mapping[self.fingerpint.class_]
        ).unsqueeze_(-1)


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | Tensor:        
        data = torch.tensor(add_channel_dim(self.volume[index]))

        if self.transformer:
            data = self.transformer(data)

        if self.phase == 'test':
            return data
        
        return (data, self._label)
    

    def __len__(self) -> int:
        return self.volume.shape[0]