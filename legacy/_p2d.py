import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from pathlib import Path

basedir_data = Path('/home/jannik/storage/wood/')


from prediction import load_model, Predictor
from loader import SliceLoader
from loadhelpers import TileDatasetBuilder, instances
from evametrics import Cardinalities
from trackers import TrackedCardinalities
from datasets import SliceDataset
from augmentations import Transformer
from transformbuilder import from_configurations


modeldir_volumetric = Path('/home/jannik/storage/trainruns-wood-volumetric/')
modeldir_planar = Path('/home/jannik/storage/trainruns-wood/')

model_2D = load_model(
    modeldir_planar / 'default-full' / 'checkpoints' / 'mdl-epoch-7.pth', dimensionality='2D'
)

IDs_int = [5, 17 ] #, 12, 20]
IDs_str = [f'CT{n}' for n in IDs_int]


training_transformer_config_2D = [
        {'name' : 'Normalize', 'mean' : 110, 'std' : 950},
        {'name' : 'Resize', 'size' : (512, 512), 'antialias' : True}
        #{'name' : 'Rotate90', 'dims' : (1, 2)}
]
transformer = Transformer(*from_configurations(training_transformer_config_2D))

slices = list(instances(IDs_str))
print(f'Loaded N = {len(slices)} slices for inference')

dataset2D = SliceDataset(phase='train', classlabel_mapping={'ahorn' : 0, 'kiefer' : 1},
                         slices=slices, transformer=transformer)


loader_2D = torch.utils.data.DataLoader(dataset2D, batch_size=64, num_workers=32)


device2D = torch.device('cuda:1')


predictor2D = Predictor(model=model_2D,
                        criterion=torch.nn.BCEWithLogitsLoss(),
                        loader=loader_2D,
                        device=device2D)


result2D = predictor2D.run()



print(result2D)







