import numpy as np
import torch

from datasets import TileDataset
from augmentations import Transformer
from transformbuilder import from_configurations

from loadhelpers import TileDatasetBuilder

def test():

    phase = 'train'
    tileshape = (256, 256, 256)
    configs = [
        {'name' : 'Normalize3D', 'mean' : 110, 'std' : 950}
    ]


    builder = TileDatasetBuilder()

    datasets = builder.build('CT10', 'CT20', 'CT10',
                             phase=phase, tileshape=tileshape,
                             transform_configurations=configs)    

    for dataset in datasets:
        print('dataset str repr :: ', dataset)
        index = 12
        item, label = dataset[index]
        print(dataset.fingerprint)
        print('length is :: ', len(dataset))
        print(f'label is {label}')
        print(item.shape)
        print(type(item))

    raise Exception

    path = '/home/jannik/storage/wood/custom/CT10.zarr'
    phase = 'train'
    tileshape = (256, 256, 256)
    configs = [
        {'name' : 'Normalize3D', 'mean' : 110, 'std' : 950}
    ]
    classlabel_mapping = {'ahorn' : 0, 'kiefer' : 1}

    transformer = Transformer(*from_configurations(configs))

    dataset = TileDataset(
        path=path, phase=phase, tileshape=tileshape, transformer=transformer,
        classlabel_mapping=classlabel_mapping
    )

    print(dataset)

    element, label = dataset[2]
    print(f'label = {label}')
    print(type(element))
    print(f'shape = {element.shape}')
    print(f'mean = {torch.mean(element)}')



def main():
    test()


if __name__ == '__main__':
    main()
