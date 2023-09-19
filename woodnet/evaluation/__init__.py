import json
import datetime
import torch

from pathlib import Path


from woodnet.custom.types import PathLike
from woodnet.datasets.planar import SliceDataset
from woodnet.datasets.volumetric import TileDataset
from woodnet.transformations import from_configurations
from woodnet.transformations.transformer import Transformer
from woodnet.tboardhelpers import retrieve_losses

from woodnet.prediction import load_model, Predictor
from woodnet.utils import create_timestamp

class IOHandler:

    def __init__(self, directory: PathLike) -> None:
        self.directory = directory

    def setup(self):
        pass



def build_dataset_2D(IDs: list[str],
                      transformer_config: list[dict]) -> SliceDataset:
    classlabel_mapping = {'ahorn' : 0, 'kiefer' : 1}
    transformer = Transformer(*from_configurations(transformer_config))
    slices = list(loadhelpers.instances(IDs))
    dataset = SliceDataset(phase='train', slices=slices,
                           classlabel_mapping=classlabel_mapping,
                           transformer=transformer)
    return dataset


def build_dataset_3D(IDs: list[str],
                      tileshape: tuple[int],
                      transformer_config: list[dict]) -> list[TileDataset]:

    builder = loadhelpers.TileDatasetBuilder()
    datasets = builder.build(*IDs, phase='train', tileshape=tileshape,
                             transform_configurations=transformer_config)
    dataset = torch.utils.data.ConcatDataset(datasets)
    return dataset




def evaluate(src_training_dir: PathLike,
             save_dir: PathLike,
             epoch: int,
             IDs: list[str],
             transformer_config: list[dict],
             device: torch.device | str,
             batch_size: int,
             tileshape: tuple[int] | None = None,
             dimensionality: str = '2D',
             num_workers: int = 32) -> None:
    """Evaluate model located in directory and write results as JSON
    to the save directory.
    """
    src_training_dir = Path(src_training_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    if dimensionality == '3D':
        if tileshape is None:
            raise RuntimeError('3D model requires tileshape specification')
        dataset = build_dataset_3D(IDs, tileshape, transformer_config)

    elif dimensionality == '2D':
        dataset = build_dataset_2D(IDs, transformer_config)

    else:
        raise ValueError('wtf up with that dimensions man')


    model_path = src_training_dir / 'checkpoints' / f'mdl-epoch-{epoch}.pth'

    device = torch.device(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    model = load_model(model_path, dimensionality=dimensionality)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=num_workers)

    predictor = Predictor(model, criterion=criterion,
                          loader=loader, device=device)
    
    loss, cardinalities = predictor.run()

    results = {
        'timestamp' : get_timestamp(),
        'modelpath' : str(model_path),
        'epoch' : epoch,
        'IDs' : IDs,
        'transformer' : transformer_config,
        'loss' : retrieve_losses(src_training_dir),
        'evaluation' : {
            'loss' : loss.value,
            'TP' : cardinalities.TP,
            'TN' : cardinalities.TN,
            'FP' : cardinalities.FP,
            'FN' : cardinalities.FN,
            'ACC' : cardinalities.ACC
        }
    }
    save_path = save_dir / f'evaluation-{get_timestamp()}.json'
    with save_path.open(mode='w') as handle:
        json.dump(results, handle)
    return results




class Evaluator:

    def __init__(self,
                 transformer_config,
                 directory: PathLike) -> None:

        self.transformer_config: list[dict] = transformer_config
        self.directory = Path(directory)




