import logging
from torch.utils.tensorboard.writer import SummaryWriter

from woodnet.directoryhandlers import ExperimentDirectoryHandler

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def init_writer(handler: ExperimentDirectoryHandler) -> SummaryWriter:
    """
    Initialize SummaryWriter instance directly from an
    ExperimentDirectoryHandler.
    """
    writer_class = SummaryWriter
    logger.info(f'Initializing new {writer_class} with target log '
                f'directory: \'{handler.logdir}\'')
    writer = writer_class(log_dir=handler.logdir)
    return writer
