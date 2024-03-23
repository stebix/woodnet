import torch
import logging
import tqdm.auto as tqdm

from collections.abc import MutableMapping, Callable
from copy import deepcopy
from typing import Literal, Protocol, TypeAlias
from torch.utils.tensorboard import SummaryWriter

from woodnet.checkpoint.registry import Registry
from woodnet.directoryhandlers import ExperimentDirectoryHandler
from woodnet.evaluation.metrics import compute_cardinalities
from woodnet.trackers import TrackedScalar, TrackedCardinalities
from woodnet.logtools.tensorboard.modelparameters.loggers import VoidLogger
from woodnet.trainer.utils import TerminationReason, get_batchsize
from woodnet.logtools.dict import LoggedDict
from woodnet.custom.exceptions import ConfigurationError
from woodnet.extent import compute_training_extent
from woodnet.logtools.tensorboard import init_writer

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)

Tensor: TypeAlias = torch.Tensor
DataLoader: TypeAlias = torch.utils.data.DataLoader


class ParameterLogger(Protocol):
    def log_weights(self, model: torch.nn.Module, iteration: int) -> None:
        pass

    def log_gradients(self, model: torch.nn.Module, iteration: int) -> None:
        pass


class Trainer:
    """
    New modern and shiny trainer.

    TODO: Currently saving model checkpoints is performed by both
    the `ExperimentDirectoryHandler` instance and the `Registry`
    instance.
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
                 score_registry: Registry,
                 device: str | torch.device,
                 max_num_epochs: int,
                 max_num_iters: int,
                 log_after_iters: int,
                 validate_after_iters: int,
                 use_amp: bool,
                 use_inference_mode: bool,
                 save_model_checkpoint_every_n: int,
                 writer: SummaryWriter,
                 parameter_logger: ParameterLogger | None
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
        self.writer = writer
        self.parameter_logger = parameter_logger or VoidLogger()

        self.validation_metric = validation_metric
        self.score_registry = score_registry

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
        self.parameter_logger = parameter_logger

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

        # report model weights and gradients for the current iteration
        self.parameter_logger.log_weights(model=self.model, iteration=self.iteration)
        self.parameter_logger.log_gradients(model=self.model, iteration=self.iteration)


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
        item = (metric_value, self.model)
        wasteitem = self.score_registry.register(item)

        if wasteitem:
            wasteitem.remove()

        return None
    

    @classmethod
    def create(cls,
               configuration: MutableMapping,
               model: torch.nn.Module | Callable,
               handler: ExperimentDirectoryHandler,
               device: torch.device,
               optimizer: torch.optim.Optimizer,
               criterion: torch.nn.Module | Callable,
               loaders: MutableMapping[str, DataLoader],
               validation_criterion: Callable | None = None,
               leave_total_progress: bool = True
               ) -> 'Trainer':
        
        # todo maybe extricate the trainer configuration        
        if 'trainer' not in configuration:
            raise ConfigurationError('missing required trainer subconfiguration')

        trainerconf = LoggedDict(deepcopy(configuration['trainer']), logger=logger)
        
        logger.info(f'Using training device: \'{device}\'')
        
        # TODO: implement systematic retrieval for multiple trainer classes
        logger.info(f'Using setting leave_total_progress: {leave_total_progress}')

        trainloader = loaders.get('train')
        batchsize = trainloader.batch_size
        # extract configuration values for extent specification
        conf_max_num_epochs = trainerconf.pop(key='max_num_epochs', default=None)
        conf_max_num_iters = trainerconf.pop(key='max_num_iters', default=None)
        conf_gradient_budget = trainerconf.pop(key='gradient_budget', default=None)

        extent = compute_training_extent(loader_length=len(trainloader), max_num_epochs=conf_max_num_epochs,
                                         max_num_iters=conf_max_num_iters, gradient_budget=conf_gradient_budget,
                                         batchsize=batchsize)
        logger.info(extent)

        log_after_iters = trainerconf.pop('log_after_iters', default=250)
        validate_after_iters = trainerconf.pop('validate_after_iters', default=1000)
        use_amp = trainerconf.pop('use_amp', default=True)
        use_inference_mode = trainerconf.pop('use_inference_mode', default=True)
        save_model_checkpoint_every_n = trainerconf.pop('save_model_checkpoint_every_n', default=5)

        if validation_criterion is None:
            validation_criterion = criterion
            logger.info(f'No validation criterion set, using deduced criterion {validation_criterion}')
        else:
            logger.info(f'Using explicitly provided validation criterion \'{validation_criterion}\'')
        
        validation_metric = trainerconf.pop(key='validation_metric', default='ACC')

        # construct the score registry
        registry_conf = trainerconf.pop('score_registry', default=None)
        
        writer = init_writer(handler=handler)    
        logger.debug(f'Injecting remaining kwargs into trainer constructor: {trainerconf}')
        
        trainer = cls(
            model=model, optimizer=optimizer, criterion=criterion,
            loaders=loaders, handler=handler, validation_criterion=validation_criterion,
            device=device, max_num_epochs=extent.max_num_epochs, max_num_iters=extent.max_num_iters,
            log_after_iters=log_after_iters, validate_after_iters=validate_after_iters,
            use_amp=use_amp, use_inference_mode=use_inference_mode,
            save_model_checkpoint_every_n=save_model_checkpoint_every_n,
            validation_metric=validation_metric,
            leave_total_progress=leave_total_progress,
            **trainerconf
        )

        return trainer
