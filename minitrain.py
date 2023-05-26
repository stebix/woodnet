import json
import torch

from collections import defaultdict
from pathlib import Path
from itertools import chain

import transformbuilder
from datasets import SliceDataset
from augmentations import Transformer
from loader import SliceLoader, LoadingStrategy
from io_handlers import IOHandler

from loader import parse_directory_identifier
from datastats import collect_data_directories
from loadhelpers import instance, get_ID_by



def taskschedule() -> dict[str, dict]:
     schedule = {
         'transversal-only' : {
            'training_ID' : [3, 5, 7, 6, 2, 14, 11, 10],
            'validation_ID' : [8, 9, 12, 13]
         },
         'fuzzy-axial-only' : {
             'training_ID' : [20, 21, 22, 16, 17, 18, 19],
             'validation_ID' : [5, 6, 7, 2, 10, 12]
         },
         'default-full-fold-2' : {
             'training_ID' : [12, 17, 5, 20, 3, 6, 8, 9, 22, 16, 17, 2, 14, 11, 19, 13],
             'validation_ID' : [7, 21, 10, 18]
         },
         'default-full-fold-3' : {
             'training_ID' : [12, 17, 5, 20, 7, 21, 10, 18, 3, 9, 22, 16, 11, 19, 10, 13],
             'validation_ID' : [2, 14, 6, 8]
         }
     }
     return schedule



def main(traindir_name, training_ID, validation_ID):



    def sort_by_wood(directories: list[Path]) -> dict[str, dict]:
        class_sorted = defaultdict(dict)
        for directory in directories:

            if 'kiefer' in str(directory).lower():
                attributes = parse_directory_identifier(directory.stem)
                ID = int(attributes['ID'].removeprefix('CT'))
                class_sorted['kiefer'][ID] = directory

            elif 'ahorn' in str(directory).lower():
                attributes = parse_directory_identifier(directory.stem)
                ID = int(attributes['ID'].removeprefix('CT'))
                class_sorted['ahorn'][ID] = directory

        return class_sorted
    

    def export_IDs(train: list, val: list, handler: IOHandler) -> None:
        split = {
            'training' : train,
            'validation' : val
        }
        path: Path = handler.dirs.log / 'ID_split.json'
        with path.open(mode='w') as handle:
            json.dump(split, handle)


    # directories
    basedir = Path('/home/jannik/storage/wood/complete/')

    datadirs = collect_data_directories(basedir)['complete']
    class_sorted = sort_by_wood(datadirs)
    ID_mapping = {**class_sorted['kiefer'], **class_sorted['ahorn']}
    

    # ahorn_train_ID = [16, 2, 14, 11, 19, 18, 10, 13]
    # ahorn_val_ID = [12, 17]

    # kiefer_train_ID = [21, 15, 7, 3, 9, 6, 8, 22]
    # kiefer_val_ID = [5, 20]1
    # chain(ahorn_train_ID, kiefer_train_ID)
    # chain(ahorn_val_ID, kiefer_val_ID)

    train_dirs = []
    for ID in training_ID:
        path = ID_mapping[ID]
        train_dirs.append(path)

    val_dirs = []
    for ID in validation_ID:
        path = ID_mapping[ID]
        val_dirs.append(path)

    print(len(train_dirs))
    print()
    print(len(val_dirs))


    # ahorn_train_dirs = [
    #     '/home/jannik/storage/wood/complete/CT10_Ahorn_40kV_200muA_5s_1mitt/',
    #     '/home/jannik/storage/wood/complete/CT11_Ahorn_40kV_200muA_5s_1mitt/',
    #     '/home/jannik/storage/wood/complete/CT12_Ahorn_40kV_200muA_5s_1mitt/'
    # ]
    # kiefer_train_dirs = [
    #     '/home/jannik/storage/wood/complete/CT20_Kiefer_40kV_200muA_5s_1mitt/',
    #     '/home/jannik/storage/wood/complete/CT21_Kiefer_40kV_200muA_5s_1mitt/',
    #     '/home/jannik/storage/wood/complete/CT22_Kiefer_40kV_200muA_5s_1mitt/'
    # ]

    # ahorn_val_dirs = [
    #     '/home/jannik/storage/wood/complete/CT10_Ahorn_40kV_200muA_5s_1mitt/',
    #     '/home/jannik/storage/wood/complete/CT11_Ahorn_40kV_200muA_5s_1mitt/',
    #     '/home/jannik/storage/wood/complete/CT12_Ahorn_40kV_200muA_5s_1mitt/'
    # ]
    # val_dirs = [
    #     '/home/jannik/storage/wood/complete/CT9_Kiefer_40kV_200muA_5s_1mitt/',
    #     '/home/jannik/storage/wood/complete/CT16_Ahorn_40kV_200muA_5s_1mitt/'
    # ]


    sliceloader = SliceLoader()
    sliceloader.strategy = LoadingStrategy.LAZY

    training_transformer_config = [
        {'name' : 'Resize', 'size' : (512, 512), 'antialias' : True},
        {'name' : 'Normalize', 'mean' : 110, 'std' : 950}
    ]

    validation_transformer_config = [
        {'name' : 'Resize', 'size' : (512, 512), 'antialias' : True},
        {'name' : 'Normalize', 'mean' : 110, 'std' : 950}
    ]

    training_transformer = Transformer(
        *transformbuilder.from_configurations(training_transformer_config)
    )
    validation_transformer = Transformer(
        *transformbuilder.from_configurations(validation_transformer_config)
    )


    train_slices = []
    for d in train_dirs:
        d = Path(d)
        train_slices.extend(
            sliceloader.from_directory(d)
        )


    val_slices = []
    for d in val_dirs:
        d = Path(d)
        val_slices.extend(
            sliceloader.from_directory(d)
        )

    # import tqdm
    # for vs in tqdm.tqdm(val_slices, unit='vslc', leave=True):
    #     for ts in train_slices:
    #         assert vs.filepath != ts.filepath, 'oh no val and train intersect'

    # print('apparently no intersection! Nioce!')

    # subselect_only_N: int = 800
    # train_slices = random.choices(train_slices, k=subselect_only_N)

    # subselect_val: int = 2500
    # val_slices = random.choices(val_slices, k=subselect_val)

    print(f'Total train slices: {len(train_slices)}')
    print(f'N_val mixture: {len(val_slices)}')



    mapping = {'ahorn' : 0, 'kiefer' : 1}

    train_dataset = SliceDataset(phase='train', slices=train_slices,
                                 classlabel_mapping=mapping,
                                 transformer=training_transformer)

    val_dataset = SliceDataset(phase='val', slices=val_slices,
                               classlabel_mapping=mapping,
                               transformer=validation_transformer)


    import models
    from training import Trainer

    model = models.ResNet18(in_channels=1)

    # training parameters
    max_num_epochs = 10
    max_num_iters = 100000
    lr = 1e-3
    batchsize = 128

    loaders = {
        'train' : torch.utils.data.DataLoader(train_dataset, batch_size=batchsize,
                                              shuffle=True, num_workers=16),
        'val' : torch.utils.data.DataLoader(val_dataset, batch_size=batchsize,
                                            shuffle=False, num_workers=16)
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device('cuda:3')

    model.to(device)

    training_dir = Path(f'/home/jannik/storage/trainruns-wood/{traindir_name}')
    handler = IOHandler(training_dir)

    export_IDs(
        train=training_ID,
        val=validation_ID,
        handler=handler
    )

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion,
                      loaders=loaders, io_handler=handler,
                      validation_criterion=criterion,
                      device=device, max_num_epochs=max_num_epochs,
                      max_num_iters=max_num_iters, log_after_iters=250,
                      validate_after_iters=250)
    
    trainer.train()


if __name__ == '__main__':
    schedule = taskschedule()
    for dirname, idmapping in schedule.items():
        print(f'targeting :: {dirname}')
        print(f'Train ID :: {idmapping["training_ID"]}')
        print(f'Validation ID :: {idmapping["training_ID"]}')

        main(dirname, **idmapping)

