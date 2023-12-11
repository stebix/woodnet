import numpy as np
import torch

from pathlib import Path
from tqdm.auto import tqdm

from models.planar import ResNet18
from models.volume import ResNet3D

from trackers import TrackedScalar, TrackedCardinalities
from evametrics import compute_cardinalities


Tensor = torch.Tensor
DataLoader = torch.utils.data.DataLoader


def get_batchsize(tensor: Tensor) -> int:
    return tensor.shape[1]


def load_model(path: str | Path, dimensionality: str = '3D') -> torch.nn.Module:
    """Quickload as long as only ResNet18 exists"""
    path = Path(path)
    model_class = ResNet18 if dimensionality == '2D' else ResNet3D
    restored_state_dict = torch.load(path)
    model = model_class(in_channels=1)
    model.load_state_dict(restored_state_dict)
    return model


class Predictor:
    """
    Encapsulate prediction tasks.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 loader: DataLoader,
                 device: str | torch.device,
                 ) -> None:
        self.model = model
        self.criterion = criterion
        self.loader = loader
        self.criterion = criterion
        if not isinstance(device, torch.device):
            self.device = torch.device(device)
        else:
            self.device = device
        

    def forward_pass(self, data: Tensor, label: Tensor,
                     criterion: torch.nn.Module) -> tuple[Tensor]:
        """Compute integrated forward pass."""
        prediction = self.model(data)
        loss = criterion(prediction, label)
        return (prediction, loss)


    def predict(self, loader: DataLoader) -> tuple:
        self.model = self.model.to(self.device)
        self.model.eval()
        device = self.device
        dtype = torch.float32
        criterion = self.criterion
        sigmoid = torch.nn.Sigmoid()
        wrapped_loader = tqdm(loader, unit='bt', desc='validation', leave=False)
        running_validation_loss = TrackedScalar()
        running_validation_metrics = TrackedCardinalities()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(wrapped_loader):
                data, label = batch_data
                data = data.to(device=device, dtype=dtype, non_blocking=True)
                label = label.to(device=device, dtype=dtype, non_blocking=True)

                prediction, loss = self.forward_pass(data, label, criterion)
                running_validation_loss.update(loss.item(), get_batchsize(data))

                prediction = sigmoid(prediction)
                cardinalities = compute_cardinalities(prediction, label)
                running_validation_metrics.update(cardinalities)

        return (running_validation_loss, running_validation_metrics)
    

    def run(self) -> tuple:
        return self.predict(self.loader)




