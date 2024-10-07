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
    Perform prediction task on a given model and loader.

    Parameters
    ----------

    model : torch.nn.Module
        Model to use for prediction.

    criterion : torch.nn.Module
        Loss function to compute total loss.

    loader : DataLoader
        DataLoader to source data from.

    device : str | torch.device
        Device to use for prediction task.

    use_amp : bool, optional
        Flag to enable automatic mixed precision.
        Default is ``True``.

    use_inference_mode : bool, optional
        Flag to enable inference mode for the model.
        If false, ``no_grad`` mode is used.
        Default is ``True``.
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
        """
        Compute forward pass and loss.

        Parameters
        ----------

        data : Tensor
            Input data tensor.

        label : Tensor
            Label tensor.

        criterion : torch.nn.Module
            Loss function to compute loss.

        Returns
        -------

        (prediction, loss) : tuple[torch.Tensor]
        """
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            prediction = self.model(data)
            loss = criterion(prediction, label)
        return (prediction, loss)
    

    def disabled_gradient_context(self):
        """Context manager for automated disabled gradient computation context"""
        if self.use_inference_mode:
            return torch.inference_mode()
        else:
            return torch.no_grad()


    def predict(self, loader: DataLoader) -> tuple:
        """
        Perform the full prediction task on the given loader.
        Running metrics are recorded and returned.

        Parameters
        ----------

        loader : DataLoader
            DataLoader to source data from.

        Returns
        -------

        (running_validation_loss, running_validation_metrics) : tuple
            Loss and metrics tracked during prediction run.
        """
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
        """Set the model to inference mode."""
        self.model.eval()
        self.model.testing = True
        self.logger.info('Configured model to inference mode')

    def set_model_training(self) -> None:
        """Set the model to training mode."""
        self.model.train()
        self.model.testing = False
        self.logger.info('Configured model to training mode')
    

    def run(self) -> tuple:
        """Run the prediction task."""
        return self.predict(self.loader)




