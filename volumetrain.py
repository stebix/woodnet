import json
import torch

from pathlib import Path

from models.volume import ResNet3D
from training import Trainer
from io_handlers import IOHandler
from loadhelpers import TileDatasetBuilder


def export_IDs(train: list, val: list, handler: IOHandler) -> None:
    split = {
        'training' : train,
        'validation' : val
    }
    path: Path = handler.dirs.log / 'ID_split.json'
    with path.open(mode='w') as handle:
        json.dump(split, handle)


def main():

    # default fold
    #training_IDs = [
    #    'CT16', 'CT2', 'CT14', 'CT11', 'CT19', 'CT18', 'CT10', 'CT13',
    #    'CT21', 'CT15', 'CT7', 'CT3', 'CT9', 'CT6', 'CT8', 'CT22'
    #]
    #validation_IDs = ['CT12', 'CT17', 'CT5', 'CT20']

    # secondary fold
    """
    training_IDs = [
        f'CT{num}'
        for num in [12, 17, 5, 20, 3, 6, 8, 9, 22, 16, 17, 2, 14, 11, 19, 13]
    ]
    validation_IDs = [
        f'CT{num}'
        for num in [7, 21, 10, 18]
    ]
    """


    # transversal only 
    training_IDs= [
        f'CT{num}'
        for num in [3, 5, 7, 6, 2, 14, 11, 10]
    ]
    validation_IDs= [
        f'CT{num}'
        for num in [8, 9, 12, 13]
    ]


    # training data settings
    tileshape = (256, 256, 256)
    training_transformer_config = [
        {'name' : 'Normalize3D', 'mean' : 110, 'std' : 950}
        #{'name' : 'Rotate90', 'dims' : (1, 2)}
    ]
    validation_transformer_config = [
        {'name' : 'Normalize3D', 'mean' : 110, 'std' : 950}
        #{'name' : 'Rotate90', 'dims' : (1, 2)}
    ]

    builder = TileDatasetBuilder()
    training_datasets = builder.build(*training_IDs, phase='train', tileshape=tileshape,
                                      transform_configurations=training_transformer_config)

    validation_datasets = builder.build(*validation_IDs, phase='train', tileshape=tileshape,
                                        transform_configurations=validation_transformer_config)

    training_dataset = torch.utils.data.ConcatDataset(training_datasets)
    validation_dataset = torch.utils.data.ConcatDataset(validation_datasets)

    print(f'Total train subvolumes: {len(training_dataset)}')
    print(f'Total val subvolumes: {len(validation_dataset)}')

    model = ResNet3D(in_channels=1)

    # training parameters
    max_num_epochs = 75
    max_num_iters = 150000
    lr = 1e-3
    batchsize = 12
    log_after_iters: int = 50
    validate_after_iters: int = 50

    loaders = {
        'train' : torch.utils.data.DataLoader(training_dataset, batch_size=batchsize,
                                              shuffle=True, num_workers=16),
        'val' : torch.utils.data.DataLoader(validation_dataset, batch_size=batchsize,
                                            shuffle=False, num_workers=16)
    }


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device('cuda:3')

    model.to(device)

    traindir_name = 'run-6-tv-only'
    training_dir = Path(f'/home/jannik/storage/trainruns-wood-volumetric/{traindir_name}')
    handler = IOHandler(training_dir)

    export_IDs(
        train=training_IDs,
        val=validation_IDs,
        handler=handler
    )

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion,
                      loaders=loaders, io_handler=handler,
                      validation_criterion=criterion,
                      device=device, max_num_epochs=max_num_epochs,
                      max_num_iters=max_num_iters,
                      log_after_iters=log_after_iters,
                      validate_after_iters=validate_after_iters)
    
    trainer.train()


if __name__ == '__main__':
    main()
