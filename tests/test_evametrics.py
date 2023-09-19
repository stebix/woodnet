import torch

from woodnet.evaluation.metrics import compute_cardinalities


def test_optimal_classification():
    batchsize = 5
    do_binarize = True
    threshold = 0.5

    prediction = torch.rand(batchsize)
    # harden via threshold
    target = torch.where(prediction > 0.5, 1, 0)
    cardinalities = compute_cardinalities(prediction, target,
                                          do_binarize=do_binarize,
                                          threshold=threshold)
    total = torch.ones(1, dtype=torch.long) * batchsize
    correct_classifications = cardinalities.TP + cardinalities.TN
    assert torch.allclose(correct_classifications, total)


def test_pessimal_classification():
    batchsize = 5
    do_binarize = True
    threshold = 0.5

    prediction = torch.rand(batchsize)
    # harden via threshold
    target = torch.where(prediction > 0.5, 0, 1)
    cardinalities = compute_cardinalities(prediction, target,
                                          do_binarize=do_binarize,
                                          threshold=threshold)
    expected_correct = torch.zeros(1, dtype=torch.long)
    correct_classifications = cardinalities.TP + cardinalities.TN
    assert torch.allclose(correct_classifications, expected_correct)