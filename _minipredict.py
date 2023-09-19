import os
import torch

import loadhelpers
from prediction import load_model, Predictor
from loader import SliceLoader, LoadingStrategy
from datasets import SliceDataset
from augmentations import Transformer
from transformbuilder import from_configurations


def main():
    os.environ['WOOD_DATA_DIRECTORY'] = '~/jannik/storage/wood'
    model = load_model(
        '/home/jannik/storage/trainruns-wood-legacy/trainruns-wood-2/birthplace7/checkpoints/mdl-epoch-7.pth',
        dimensionality='2D')

    IDs = ['CT12', 'CT17', 'CT5', 'CT20']
    classlabel_mapping = {'ahorn' : 0, 'kiefer' : 1}

    configurations = [
        {'name' : 'Resize', 'size' : (512, 512), 'antialias' : True},
        {'name' : 'Normalize', 'mean' : 110, 'std' : 950},
        {'name' : 'GaussianBlur', 'kernel_size' : 5, 'sigma' : (0.6, 0.8)},
        {'name' : 'RandomRotation', 'degrees' : (-35, +35)},
    ]
    transformer = Transformer(*from_configurations(configurations))

    loader = SliceLoader()
    loader.strategy = LoadingStrategy.LAZY

    slices = list(loadhelpers.instances(IDs))
    print(f'Loaded N = {len(slices)} slices for inference')
    dataset = SliceDataset(phase='train', slices=slices,
                           classlabel_mapping=classlabel_mapping,
                           transformer=transformer)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=128, num_workers=64
    )


    criterion = torch.nn.BCEWithLogitsLoss()
    predictor = Predictor(model, criterion, dataloader, device='cuda:3')

    loss, cards = predictor.run()

    print(f'Cards .. {cards}')
    print(f'Predictor exitet with loss :: {loss.value:.5f}')
    print('finito')




if __name__ == '__main__':
    main()
    print('exit')
