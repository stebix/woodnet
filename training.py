import torch

from tqdm.auto import tqdm
from typing import Callable, Iterable, Union, Literal

from evametrics import compute_cardinalities
from trackers import TrackedScalar, TrackedCardinalities

Tensor = torch.Tensor
DataLoader = torch.utils.data.DataLoader


def validate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             validationloader: DataLoader,
             device: torch.device = 'cpu',
             dtype: torch.dtype = torch.float32) -> float:
    """Validate model, duh."""
    model.eval()
    validation_loss_history = []
    wrapped_loader = tqdm(validationloader, unit='bt',
                          desc='ValLoader', leave=False)
    with torch.no_grad():
        for batch in wrapped_loader:
            data, label = batch
            data = data.to(device=device, dtype=dtype)
            label = label.to(device=device, dtype=dtype)

            prediction = model(data)
            loss = criterion(prediction, label)

            validation_loss_history.append(loss.item())
    
    return validation_loss_history




def train(model: torch.nn.Module,
          train_iters: int,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          trainloader: DataLoader,
          validationloader: DataLoader,
          validate_every_n_epoch: int,
          dtype: torch.dtype = torch.float32,
          device: torch.device = 'cpu') -> None:


    train_iter = 0
    current_epoch = 0

    train_loss_history = []

    # torch setup
    model.train(True)
    device = torch.device(device)
    total_pbar = tqdm(total=train_iters, unit='iter', desc='TrainIter')

    while train_iter <= train_iters:
        # epoch happens inside this loop
        for batch in tqdm(trainloader, unit='bt', desc='LoaderIter', leave=False):
            # data moving
            data, label = batch
            data = data.to(device=device, dtype=dtype, non_blocking=True)
            label = label.to(device=device, dtype=dtype, non_blocking=True)
            # actual deep learning
            optimizer.zero_grad()
            prediction = model(data)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            # bookkeeping
            train_loss_history.append(loss.item())
            total_pbar.update()

        current_epoch += 1

        if current_epoch % validate_every_n_epoch == 0:
            validate(model, validationloader)


import logging
from typing import Protocol, Type

class IOHandler(Protocol):
    pass


from torch.utils.tensorboard import SummaryWriter

def get_batchsize(tensor: Tensor) -> int:
    return tensor.shape[1]


class Trainer:

    writer_class: Type[SummaryWriter] = SummaryWriter

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 loaders: dict[str, DataLoader],
                 metrics: Iterable[Callable[..., Tensor]],
                 io_handler: IOHandler,
                 validation_criterion: torch.nn.Module,
                 device: Union[str, torch.device],
                 max_num_epochs: int,
                 max_num_iters: int,
                 log_after_iters: int,
                 validate_after_iters: int
                 ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        if not isinstance(device, torch.device):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.train_loader = loaders.get('train')
        self.val_loader = loaders.get('validation')

        self.metrics = metrics
        self.validation_criterion = validation_criterion
        self.io_handler = io_handler
        # writer receives its log directory destination from
        # io_handler instance
        self.writer = self._init_writer()

        # bookkeeping
        self.max_num_epochs = max_num_epochs
        self.max_num_iters = max_num_iters
        self.epoch: int = 0
        self.iteration: int = 1

        self.running_train_loss = TrackedScalar()
        self.running_train_metrics = TrackedCardinalities()

        self.log_after_iters = log_after_iters
        self.validate_after_iters = validate_after_iters


    def train(self) -> None:
        loader = self.train_loader
        for _ in range(self.max_num_epochs):
            terminated = self.train_single_epoch(loader)

            if terminated:
                print('Trainer terminated')
                return None

            self.epoch += 1

        return None
    

    def forward_pass(self, data: Tensor, label: Tensor,
                     criterion: torch.nn.Module) -> tuple[Tensor]:
        """Compute integrated forward pass."""
        prediction = self.model(data)
        loss = criterion(prediction, label)
        return (prediction, loss)
    

    def train_single_epoch(self, loader: DataLoader) -> bool:
        self.model.train()
        criterion = self.criterion
        device = self.device
        optimizer = self.optimizer
        dtype = torch.float32
        wrapped_loader = tqdm(loader, unit='bt', desc='loader', leave=False)

        for batch_idx, batch_data in enumerate(wrapped_loader):
            # data moving
            data, label = batch_data
            data = data.to(device=device, dtype=dtype, non_blocking=True)
            label = label.to(device=device, non_blocking=True)
            # actual deep learning
            optimizer.zero_grad()
            prediction, loss = self.forward_pass(data, label, criterion)
            loss.backward()
            optimizer.step()

            self.running_train_loss.update(loss.item(), get_batchsize(data))

            if self.iteration % self.log_after_iters == 0:
                self.on_log_iteration(prediction, label)

            if self.iteration % self.validate_after_iters == 0:
                self.on_validation_iteration()

            # bookkeeping
            self.iteration += 1

            if self.should_terminate():
                return True
        
        self.running_train_loss.reset()
        self.running_train_metrics.reset()
        return False


    def should_terminate(self) -> bool:
        if self.iteration >= self.max_num_iters:
            return True
        return False
    

    def on_log_iteration(self, prediction: Tensor, label: Tensor) -> None:
        # report loss
        loss_tag = f'loss/training_{self.criterion.__class__.__name__}'
        self.writer.add_scalar(loss_tag, self.running_train_loss.value,
                               global_step=self.iteration)
        # update running metrics with current prediction and label
        with torch.no_grad():
            # TODO: maybe factor to model
            prediction = torch.nn.Sigmoid(prediction)
            cardinalities = compute_cardinalities(prediction, label)
            self.running_train_metrics(cardinalities)
        
        # report training metrics
        self.log_tracked_cardinalities(self.running_train_metrics, 'train')


    def log_tracked_cardinalities(self,
                                  tracked_cardinalities: TrackedCardinalities,
                                  phase: Literal['train', 'val']) -> None:
        
        phase = 'training' if phase == 'train' else 'validation'
        for name in tracked_cardinalities.joint_identifiers:
            tag = f'metrics/{phase}_{name}'
            value = getattr(tracked_cardinalities, name)
            self.writer.add_scalar(tag, value, global_step=self.iteration)


    def on_validation_iteration(self) -> None:
        self.model.eval()
        self.validate(self.val_loader)
        self.model.train()


    def validate(self, loader: DataLoader) -> None:
        """Evaluate performance via validation data."""
        device = self.device
        dtype = torch.float32
        criterion = self.validation_criterion
        wrapped_loader = tqdm(loader, unit='bt', desc='val')
        running_validation_loss = TrackedScalar()
        running_validation_metrics = TrackedCardinalities()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(wrapped_loader):
                data, label = batch_data
                data = data.to(device=device, dtype=dtype, non_blocking=True)
                label = label.to(device=device, non_blocking=True)

                prediction, loss = self.forward_pass(data, label, criterion)
                running_validation_loss.update(loss.item(), get_batchsize(data))

                prediction = torch.nn.Sigmoid(prediction)
                cardinalities = compute_cardinalities(prediction, label)
                running_validation_metrics(cardinalities)
        
        # report results
        loss_tag = f'loss/validation_{criterion.__class__.__name__}'
        self.writer.add_scalar(loss_tag, running_validation_loss.value,
                               global_step=self.iteration)
        self.log_tracked_cardinalities(running_validation_metrics, 'val')


    def _init_writer(self) -> SummaryWriter:
        return self.writer_class(log_dir=self.io_handler.logs)




