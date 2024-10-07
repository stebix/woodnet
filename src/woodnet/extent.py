"""
Provide tools to gauge training extent from user-set varaibles.

@jsteb 2023
"""
import dataclasses
import enum
import math

class EstimatedTrainingExtent(enum.Enum):
    MAX_EPOCH_BOUND = 'max_epoch_bound'
    MAX_ITER_BOUND = 'max_iter_bound'
    GRADIENT_BUDGET_BOUND = 'gradient_budget_bound'


@dataclasses.dataclass
class TrainingExtent:
    extent_bound: EstimatedTrainingExtent
    actual_epochs: float
    actual_iterations: int
    total_gradient_vector_count: int
    max_num_iters: int
    max_num_epochs: int


def compute_training_extent(loader_length: int,
                            max_num_epochs: int | None,
                            max_num_iters: int | None,
                            gradient_budget: int | None,
                            batchsize: int) -> TrainingExtent:
    
    if not any((max_num_epochs, max_num_iters, gradient_budget)): 
        raise ValueError(f'training extent specification requires value for at least one variable of '
                         f'max_num_epochs, max_num_iters or gradient_budget!')
    
    if gradient_budget:
        if max_num_epochs or max_num_iters:
            raise ValueError('Not-None values for max_num_epochs or max_num_iters are '
                             'disallowed for extent bound specification via gradient_budget')
        
        bound = EstimatedTrainingExtent.GRADIENT_BUDGET_BOUND
        actual_iterations = round(gradient_budget / batchsize)
        actual_epochs = actual_iterations / loader_length
        max_num_iters = actual_iterations
        max_num_epochs = max(1, math.ceil(actual_epochs))
        total_gradient_vector_count = gradient_budget
        assert actual_iterations > 1, 'requiring at least a single iteration'
        
    else:
        iter_epoch_limit = loader_length * max_num_epochs if max_num_epochs else float('inf')
        iter_maxiter_limit = max_num_iters or float('inf')
    
        if iter_epoch_limit > iter_maxiter_limit:
            bound = EstimatedTrainingExtent.MAX_ITER_BOUND
            actual_epochs = max_num_iters / loader_length
            actual_iterations = max_num_iters
            total_gradient_vector_count = actual_iterations * batchsize 
        else:
            bound = EstimatedTrainingExtent.MAX_EPOCH_BOUND
            actual_epochs = max_num_epochs
            actual_iterations = iter_epoch_limit
            total_gradient_vector_count = actual_iterations * batchsize
    
    infodict = {
        'extent_bound' : bound,
        'actual_epochs' : actual_epochs,
        'actual_iterations' : actual_iterations,
        'total_gradient_vector_count' : total_gradient_vector_count,
        'max_num_iters' : max_num_iters or max(1, round(actual_iterations)),
        'max_num_epochs' : max_num_epochs or max(1, round(actual_epochs))
    }
    return TrainingExtent(**infodict)
