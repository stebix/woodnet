import logging
import torch
import tqdm.auto as tqdm

from typing import Literal, TypeAlias
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


from woodnet.directoryhandlers import ExperimentDirectoryHandler
from woodnet.evaluation.metrics import compute_cardinalities
from woodnet.trackers import TrackedCardinalities, TrackedScalar
from woodnet.trainer.utils import TerminationReason, get_batchsize

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)

Tensor: TypeAlias = torch.Tensor
DataLoader: TypeAlias = torch.utils.data.DataLoader

class LegacyTrainer:
    """
    Legacy-styled trainer class.
    Does not feature top-k model registry and model parameter logging.
    """
    dtype: torch.dtype = torch.float32
    leave_total_progress: bool = True

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 loaders: dict[str, DataLoader],
                 handler: ExperimentDirectoryHandler,
                 validation_criterion: torch.nn.Module,
                 validation_metric: str,
                 device: str | torch.device,
                 max_num_epochs: int,
                 max_num_iters: int,
                 log_after_iters: int,
                 validate_after_iters: int,
                 use_amp: bool,
                 use_inference_mode: bool,
                 save_model_checkpoint_every_n: int,
                 validation_metric_higher_is_better: bool
                 ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        if not isinstance(device, torch.device):
            self.device = torch.device(device)
        else:
            self.device = device

        self.train_loader = loaders.get('train')
        self.val_loader = loaders.get('val')

        self.validation_criterion = validation_criterion
        self.handler = handler
        # writer receives its log directory destination from
        # io_handler instance
        self.writer = self._init_writer()

        self.validation_metric = validation_metric
        self.validation_metric_higher_is_better = validation_metric_higher_is_better
        self.curropt_validation_metric_value = self.init_curropt_validation_metric_value()

        # bookkeeping
        self.max_num_epochs = max_num_epochs
        self.max_num_iters = max_num_iters
        self.epoch: int = 0
        self.iteration: int = 1

        self.running_train_loss = TrackedScalar()
        self.running_train_metrics = TrackedCardinalities()

        self.log_after_iters = log_after_iters
        self.validate_after_iters = validate_after_iters
        self.save_model_checkpoint_every_n = save_model_checkpoint_every_n

        self.use_amp = use_amp
        self.use_inference_mode = use_inference_mode
        self.gradscaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.logger = logging.getLogger('.'.join((LOGGER_NAME, __class__.__name__)))

        self.logger.info(f'Moving model to device: \'{self.device}\'')
        self.model = self.model.to(device=self.device, dtype=self.dtype)


    def train(self) -> None:
        loader = self.train_loader
        self.total_progress = tqdm.tqdm(
            total=self.max_num_iters, unit='it', desc='total iterations',
            leave=self.leave_total_progress, postfix=dict(epoch=self.epoch)
        )
        self.logger.info('Starting epoch training loop')
        for _ in range(self.max_num_epochs):
            terminated, reason = self.train_single_epoch(loader)

            if terminated:
                self.logger.info(f'Trainer loop concluded due to reason {reason}')
                return None

            self.epoch += 1
            self.total_progress.set_postfix_str(f'epoch={self.epoch}')

            if self.epoch % self.save_model_checkpoint_every_n == 0:
                fname = f'mdl-epoch-{self.epoch}.pth'
                self.handler.save_model_checkpoint(self.model, fname)

        self.logger.info('Exiting training method')
        return None


    def forward_pass(self, data: Tensor, label: Tensor,
                     criterion: torch.nn.Module) -> tuple[Tensor]:
        """Compute integrated forward pass."""
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            prediction = self.model(data)
            loss = criterion(prediction, label)
        return (prediction, loss)


    def train_single_epoch(self, loader: DataLoader) -> tuple[bool, TerminationReason | None]:
        self.model.train()
        criterion = self.criterion
        device = self.device
        optimizer = self.optimizer
        dtype = self.dtype
        wrapped_loader = tqdm.tqdm(loader, unit='bt', desc='loader progress', leave=False)
        self.logger.debug(f'Starting epoch {self.epoch}')
        for batch_idx, batch_data in enumerate(wrapped_loader):
            # data moving
            data, label = batch_data
            data = data.to(device=device, dtype=dtype, non_blocking=True)
            label = label.to(device=device, dtype=dtype, non_blocking=True)

            # actual deep learning
            optimizer.zero_grad()
            prediction, loss = self.forward_pass(data, label, criterion)

            self.gradscaler.scale(loss).backward()
            self.gradscaler.step(optimizer)
            self.gradscaler.update()

            self.running_train_loss.update(loss.item(), get_batchsize(data))

            if self.iteration % self.log_after_iters == 0:
                self.on_log_iteration(prediction, label)

            if self.iteration % self.validate_after_iters == 0:
                self.on_validation_iteration()

            # bookkeeping
            self.iteration += 1
            self.total_progress.update()

            terminate, reason = self.check_termination()
            if terminate:
                return (True, reason)

        self.running_train_loss.reset()
        self.running_train_metrics.reset()
        return (False, None)


    def check_termination(self) -> tuple[bool, TerminationReason | None]:
        if self.iteration >= self.max_num_iters:
            return (True, TerminationReason.MAX_ITERATIONS_REACHED)
        return (False, None)


    def on_log_iteration(self, prediction: Tensor, label: Tensor) -> None:
        # report loss
        loss_tag = f'loss/training_{self.criterion.__class__.__name__}'
        self.writer.add_scalar(loss_tag, self.running_train_loss.value,
                               global_step=self.iteration)

        # update running metrics with current prediction and label
        with self.disabled_gradient_context():

            if hasattr(self.model, 'final_nonlinearity'):
                prediction = self.model.final_nonlinearity(prediction)

            cardinalities = compute_cardinalities(prediction, label)
            self.running_train_metrics.update(cardinalities)

        # report training metrics
        self.log_tracked_cardinalities(self.running_train_metrics, 'train')


    def log_tracked_cardinalities(self,
                                  tracked_cardinalities: TrackedCardinalities,
                                  phase: Literal['train', 'val']) -> None:

        phase = 'training' if phase == 'train' else 'validation'
        for name in tracked_cardinalities.joint_identifiers:
            tag = f'{phase}_metrics/{name}'
            value = getattr(tracked_cardinalities, name)
            self.writer.add_scalar(tag, value, global_step=self.iteration)


    def on_validation_iteration(self) -> None:
        self.model.eval()
        self.validate(self.val_loader)
        self.model.train()


    def disabled_gradient_context(self):
        if self.use_inference_mode:
            return torch.inference_mode()
        return torch.no_grad()


    def validate(self, loader: DataLoader) -> None:
        """Evaluate performance via validation data."""
        device = self.device
        dtype = self.dtype
        criterion = self.validation_criterion
        wrapped_loader = tqdm.tqdm(loader, unit='bt', desc='validation', leave=False)
        running_validation_loss = TrackedScalar()
        running_validation_metrics = TrackedCardinalities()
        self.logger.debug(f'Entering validation loop')

        with self.disabled_gradient_context():
            for batch_idx, batch_data in enumerate(wrapped_loader):
                data, label = batch_data
                data = data.to(device=device, dtype=dtype, non_blocking=True)
                label = label.to(device=device, dtype=dtype, non_blocking=True)

                prediction, loss = self.forward_pass(data, label, criterion)
                running_validation_loss.update(loss.item(), get_batchsize(data))

                if hasattr(self.model, 'final_nonlinearity'):
                    prediction = self.model.final_nonlinearity(prediction)

                cardinalities = compute_cardinalities(prediction, label)
                running_validation_metrics.update(cardinalities)

        # report results
        loss_tag = f'loss/validation_{criterion.__class__.__name__}'
        self.writer.add_scalar(loss_tag, running_validation_loss.value,
                               global_step=self.iteration)
        self.log_tracked_cardinalities(running_validation_metrics, 'val')
        self.logger.debug(f'Concluded validation run with loss: {running_validation_loss.value}')

        # save optimal model checkpoint if validation metric value is optimal
        metric_value = getattr(running_validation_metrics, self.validation_metric)
        self.logger.debug(
            f'Concluded validation run with primary metric '
            f'\'{self.validation_metric}\' = {metric_value}'
        )
        if self.validation_metric_higher_is_better:
            if metric_value > self.curropt_validation_metric_value:
                self.logger.info('Saving new validation max-optimal model checkpoint')
                self.handler.save_model_checkpoint(self.model, 'optimal.pth', allow_overwrite=True)
                self.curropt_validation_metric_value = metric_value
                self.logger.debug(f'Set new optimal metric value: {metric_value}')
        else:
            if metric_value < self.curropt_validation_metric_value:
                self.logger.info('Saving new validation min-optimal model checkpoint')
                self.handler.save_model_checkpoint(self.model, 'optimal.pth', allow_overwrite=True)
                self.curropt_validation_metric_value = metric_value
                self.logger.debug(f'Set new optimal metric value: {metric_value}')


    def _init_writer(self) -> SummaryWriter:
        return self.writer_class(log_dir=self.handler.logdir)


    def init_curropt_validation_metric_value(self) -> float:
        if self.validation_metric_higher_is_better:
            return float('-inf')
        return float('+inf')
    

    @classmethod
    def create(cls, *args, **kwargs) -> 'LegacyTrainer':
        raise NotImplementedError()