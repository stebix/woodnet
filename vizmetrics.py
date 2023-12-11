import matplotlib.pyplot as plt

from typing import Protocol


class CardinalityLike(Protocol):
    TP: int
    TN: int
    FP: int
    FN: int
    


def confusion_matrix(cardinalities: CardinalityLike):
    """Plot confusion matrix from cardinalities."""
    pass



def plot_metrics(*metrics):
    """Plot metrics like ACC, F1 etc. ins ingle plot"""
    name_color_mapping = {'ACC' : 'greed', 'TPR' : 'blue', 'TNR' : 'red'}
    
    fig, ax = plt.subplots()
    ax.set_xlabel('prediction index')
    ax.set_ylabel('metric value')
    for i, metric in enumerate(metrics):
        for name, color in name_color_mapping.items():
            value = getattr(metric, name)
            ax.plot(i, value, marker='x', color=color, ls='')
    
    plt.tight_layout()




import torch
from pathlib import Path

from models import ResNet18
from loader import SliceLoader
import loadhelpers
from datasets import SliceDataset
from prediction import Predictor

from augmentations import Transformer
from transformbuilder import from_configurations

BASE_DIR = Path('/home/jannik/storage/trainruns-wood/')


def multievaluate(traindir_name: str,
                  model_epoch: int,
                  IDs: list[str],
                  transformers_configs: dict[list[dict]],
                  batch_size: int = 128):

    """Evaluate model with datasets/IDs and save loss and cardinality metrics"""
    traindir = BASE_DIR / traindir_name
    modelpath = traindir / 'checkpoints' / f'mdl-epoch-{model_epoch}.pth'
    assert modelpath.is_file(), f'Could not locate model @ "{modelpath}"'
    model = ResNet18(in_channels=1)
    model.load_state_dict(torch.load(modelpath))

    if all(isinstance(ID, int) for ID in IDs):
        IDs = [f'CT{ID}' for ID in IDs]

    transformers = {
        name : Transformer(*from_configurations(configurations))
        for name, configurations in transformers_configs.items()
    }

    mapping = {'ahorn' : 0, 'kiefer' : 1}

    slices = list(loadhelpers.instances(IDs))

    print(f'Loaded N = {len(slices)} slices for inference')


    multievaluation_results = {}

    for i, (name, transformer) in enumerate(transformers.items()):
        print(f'Transformer states {i} / {len(transformers)}')

        dataset = SliceDataset(
            phase='train', slices=slices, classlabel_mapping=mapping,
            transformer=transformer
            
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=64
        )

        criterion = torch.nn.BCEWithLogitsLoss()
        predictor = Predictor(model, criterion, loader, device='cuda:3')

        loss, metrics = predictor.run()

        multievaluation_results[name] = (loss, metrics)

    return multievaluation_results






