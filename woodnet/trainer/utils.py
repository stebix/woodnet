import enum
from torch import Tensor

class TerminationReason(enum.Enum):
    MAX_ITERATIONS_REACHED = 'max_iterations_reached'
    MAX_EPOCHS_REACHED = 'max_epochs_reached'
    MIN_LEARNING_RATE_REACHED = 'min_learning_rate_reached'


def get_batchsize(tensor: Tensor) -> int:
    return tensor.shape[1]
