import sys
import atexit
import logging

from collections.abc import Callable



def build_logintercepted_excepthook(logger: logging.Logger,
                                    previous_excepthook: Callable) -> Callable:
    """
    Build an excepthook that logs unhandled exceptions to the provided logger.
    Any actions defined by the previous_excepthook callable are then executed
    subsequently.
    """

    def logintercepted_excepthook(type, value, traceback):
        """Log some exception info and set global program state to failed."""
        import traceback as tblib

        formatted_traceback = ''.join(tblib.format_tb(traceback))
        logger.fatal(f'Run exited with unhandled exception: {type} :: \'{value}\'')
        logger.fatal(f'Traceback :: {formatted_traceback}')
        logger.fatal('Exiting, sorry sir o7 !')
        previous_excepthook(type, value, traceback)

    return logintercepted_excepthook


def install_loginterceptor_excepthook(logger: logging.Logger):
    """
    Install excepthook that logs the unhandled top level exception
    before the interpreter exits.
    """
    previous_excepthook = sys.excepthook
    sys.excepthook = build_logintercepted_excepthook(logger, previous_excepthook)
