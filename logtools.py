import pathlib
import logging
import logging.handlers


DEFAULT_FORMAT: str = '%(asctime)s - %(name)s | %(levelname)s : %(message)s'
DEFAULT_CAPACITY: int = int(1e4)

# standard library logging level
Level = int | str



def create_instance_logger(obj) -> logging.Logger:
    name = '.'.join(('main', __name__, obj.__class__.__name__))
    return logging.getLogger(name)


def create_logging_infrastructure(level: Level,
                                  streamhandler_level: Level = logging.ERROR,
                                  memoryhandler_level: Level = logging.DEBUG
                                  ) -> tuple[logging.Logger, logging.StreamHandler,
                                             logging.handlers.MemoryHandler]:
    """
    Create the basal logger infrastructure.
    The instance employs a stream handler and a memory handler. The
    latter can be utilized to flush log messages to disk as soon
    as information about the log file location is gleaned.
    """
    logger = logging.getLogger('main')
    logger.setLevel(level)

    formatter = logging.Formatter(fmt=DEFAULT_FORMAT)

    # Both handlers collect logging information but do not yet
    # require information about the log file location.
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(streamhandler_level)
    memoryhandler = logging.handlers.MemoryHandler(
        capacity=DEFAULT_CAPACITY, flushLevel=logging.ERROR,
        flushOnClose=True
    )
    memoryhandler.setLevel(memoryhandler_level)
    handlers = [streamhandler, memoryhandler]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return (logger, *handlers)



def finalize_logging_infrastructure(logger: logging.Logger,
                                    memoryhandler: logging.handlers.MemoryHandler,
                                    logfile_path: pathlib.Path) -> None:
    """
    Finalize logger infrastructure by adding a file handler and flush-removing
    the memory handler.
    """
    formatter = logging.Formatter(fmt=DEFAULT_FORMAT)
    # set up new file handler
    filehandler = logging.FileHandler(filename=logfile_path, mode='a')
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    # transfer info to file handler and remove to avoid double logging
    memoryhandler.setTarget(filehandler)
    memoryhandler.flush()
    memoryhandler.close()
    logger.removeHandler(memoryhandler)


    