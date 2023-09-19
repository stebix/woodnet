import numpy as np
import torch

from pathlib import Path
from tqdm.auto import tqdm

from woodnet.models.planar import ResNet18
from woodnet.models.volumetric import ResNet3D

from woodnet.trackers import TrackedScalar, TrackedCardinalities
from woodnet.evaluation.metrics import compute_cardinalities


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
                 use_amp: bool = True,
                 use_inference_mode: bool = True
                 ) -> None:
        self.model = model
        self.criterion = criterion
        self.loader = loader
        self.criterion = criterion
        if not isinstance(device, torch.device):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.use_amp = use_amp
        self.use_inference_mode = use_inference_mode
        

    def forward_pass(self, data: Tensor, label: Tensor,
                     criterion: torch.nn.Module) -> tuple[Tensor]:
        """Compute integrated forward pass."""
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            prediction = self.model(data)
            loss = criterion(prediction, label)
        return (prediction, loss)
    

    def disabled_gradient_context(self):
        if self.use_inference_mode:
            return torch.inference_mode()
        else:
            return torch.no_grad()


    def predict(self, loader: DataLoader) -> tuple:
        self.model = self.model.to(self.device)
        self.model.eval()
        device = self.device
        dtype = torch.float32
        criterion = self.criterion
        sigmoid = torch.nn.Sigmoid()
        wrapped_loader = tqdm(loader, unit='bt', desc='validation progress', leave=False)
        running_validation_loss = TrackedScalar()
        running_validation_metrics = TrackedCardinalities()

        with self.disabled_gradient_context():
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
    

    def set_model_inference(self) -> None:
        self.model.eval()
        self.model.testing = True
        self.logger.info('Configured model to inference mode')

    def set_model_training(self) -> None:
        self.model.train()
        self.model.testing = False
        self.logger.info('Configured model to training mode')
    

    def run(self) -> tuple:
        return self.predict(self.loader)




