import torch

import dataclasses
from typing import Optional

Tensor = torch.Tensor

@dataclasses.dataclass
class Cardinalities:
    """
    Data struct holding the four basic cardinalities of the confusion matrix.
    TP = true positive, TN = true negative, FP = false positive, FN = false negative
    """
    TP: Tensor
    TN: Tensor
    FP: Tensor
    FN: Tensor
    binarized: bool = True
    threshold: Optional[float] = None



def binarize(*tensors: Tensor, threshold: float) -> Tensor:
    return tuple(torch.where(t > threshold, 1, 0) for t in tensors)


def compute_true_positive(prediction: Tensor, target: Tensor,
                          do_binarize: bool = True, threshold: float = 0.5) -> Tensor:
    if do_binarize:
        prediction, target = binarize(prediction, target, threshold=threshold)
    return torch.sum(prediction * target)


def compute_true_negative(prediction: Tensor, target: Tensor,
                          do_binarize: bool = True, threshold: float = 0.5) -> Tensor:
    if do_binarize:
        prediction, target = binarize(prediction, target, threshold=threshold)
    tn = (1 - prediction) * (1 - target)
    return torch.sum(tn)


def compute_false_positive(prediction: Tensor, target: Tensor,
                           do_binarize: bool = True, threshold: float = 0.5) -> Tensor:
    if do_binarize:
        prediction, target = binarize(prediction, target, threshold=threshold)
    fp = prediction * (1 - target)
    return torch.sum(fp)


def compute_false_negative(prediction: Tensor, target: Tensor,
                           do_binarize: bool = True, threshold: float = 0.5) -> Tensor:
    if do_binarize:
        prediction, target = binarize(prediction, target, threshold=threshold)
    fn = (1- prediction) * target
    return torch.sum(fn)


def compute_cardinalities(prediction: Tensor, target: Tensor,
                          do_binarize: bool = True, threshold: float = 0.5
                          ) -> Cardinalities:
    """Compute all cardinalities of the confusion matrix in one go."""
    results = {
        'binarized' : do_binarize, 'threshold' : threshold if binarize else None,
        'TP' : compute_true_positive(prediction, target, do_binarize, threshold),
        'TN' : compute_true_negative(prediction, target, do_binarize, threshold),
        'FP' : compute_false_positive(prediction, target, do_binarize, threshold),
        'FN' : compute_false_positive(prediction, target, do_binarize, threshold)
    }
    return Cardinalities(**results)


def correct_classification():
    pass