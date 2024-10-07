import torch
from pathlib import Path

from evaluation import evaluate


def main():
    
    #IDs_int = [5, 17, 12, 20]
    IDs_int = [20, 22, 16, 17]
    # transversal only validation set
    IDs_int = [8, 9, 12, 13]

    IDs_int = [9, 12, 20, 21, 22, 16, 17, 18, 19]


    IDs_str = [f'CT{n}' for n in IDs_int]

    transformer_config = [
            {'name' : 'Normalize', 'mean' : 110, 'std' : 950},
            {'name' : 'Resize', 'size' : (512, 512), 'antialias' : True}
    ]
    """
    transformer_config = [
            {'name' : 'Normalize3D', 'mean' : 110, 'std' : 950}
            #{'name' : 'Resize', 'size' : (512, 512), 'antialias' : True}
    ]
    """

    dir_planar = Path('/home/jannik/storage/trainruns-wood/')
    expdir = dir_planar / 'transversal-only'
    savedir = Path('/home/jannik/storage/predictionruns/wood')

    # settings
    batch_size = 64
    device = torch.device('cuda:1')
    num_workers = 32
    tileshape = (256, 256, 256)

    results = evaluate(expdir, savedir, epoch=9, IDs=IDs_str,
                       transformer_config=transformer_config,
                       device=device, batch_size=batch_size,
                       tileshape=tileshape, dimensionality='2D',
                       num_workers=num_workers)
    print(results)


if __name__ == '__main__':
    print('Executing main ')
    main()