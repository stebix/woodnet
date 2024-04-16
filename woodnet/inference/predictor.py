"""
Programmatic interaction with prediction tasks in the context of robustness
experiments.

@jsteb 2024
"""
import contextlib
import logging
from collections.abc import Sequence

import torch
import torch.utils
import tqdm

from woodnet.inference.evaluate import evaluate
from woodnet.inference.inference import ParametrizationsContainer


LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


class Predictor:

    def __init__(self,
                 model: torch.nn.Module,
                 device: str | torch.device,
                 dtype: torch.dtype,
                 use_amp: bool,
                 use_inference_mode: bool,
                 display_transforms_progress: bool,
                 display_parametrizations_progress: bool,
                 display_loader_progress: bool,
                 leave_transforms_progress: bool = True
                 ) -> None:

        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.use_amp = use_amp
        self.use_inference_mode = use_inference_mode
        self.display_transforms_progress = display_transforms_progress
        self.display_parametrizations_progress = display_parametrizations_progress
        self.display_loader_progress = display_loader_progress
        self.leave_transforms_progress = leave_transforms_progress


    def disabled_gradient_context(self):
        if self.use_inference_mode:
            return torch.inference_mode()
        return torch.no_grad()


    def amp_context(self):
        return torch.autocast(device_type=self.device.type, enabled=self.use_amp)


    def inference_contexts(self) -> tuple:
        """Get configured no-gradient and mixed precision contexts."""
        return (self.disabled_gradient_context(), self.amp_context())


    def configure_model(self) -> None:
        """Jointly adjust model float precision, eval mode and testing flag."""
        self.model.eval()
        logger.debug('Set model to evaluation mode.')
        self.model.testing = True
        final_nonlinearity = getattr(self.model, 'final_nonlinearity', None)
        logger.debug(
            f'Set model testing flag. Final nonlinearity is: {final_nonlinearity}'
        )
        self.model.to(dtype=self.dtype, device=self.device)
        logger.debug(f'Moved model to precision {self.dtype} and device {self.device}')


    def predict(self,
                loader: torch.utils.data.DataLoader,
                transforms: Sequence[ParametrizationsContainer]) -> dict:
        """
        Perform full prediction run (full loader) for a number of parmetrizations

        Results are reported as a metadata-enhanced dictionary with
        """
        report = []
        self.configure_model()

        if self.display_transforms_progress:
            settings = {'desc' : 'transformations', 'unit' : 'tf',
                        'leave' : self.leave_transforms_progress}
            transforms = tqdm.tqdm(transforms, **settings)

        with contextlib.ExitStack() as stack:
            # conditionally use amp and inference or no-grad mode for speed/throughput
            [stack.enter_context(cm) for cm in self.inference_contexts()]
            for parametrizations in transforms:
                if self.display_transforms_progress:
                    transforms.set_postfix_str(f'transform=\'{parametrizations.name}\'')

                result = evaluate(self.model,
                                  loader=loader, parametrizations=parametrizations,
                                  device=self.device, dtype=self.dtype,
                                  display_parametrizations_progress=self.display_transforms_progress,
                                  display_loader_progress=self.display_loader_progress)

                parametrization_report = {'metadata' : parametrizations.info(), 'report' : result}
                report.append(parametrization_report)

        return report