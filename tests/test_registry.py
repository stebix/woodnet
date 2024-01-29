import logging
import pytest
import torch
import numpy as np

from functools import partial

from woodnet.checkpoint.handlers import RWDHandler
from woodnet.checkpoint.registry import Registry, ScorePreference


def init(module, value):
    if type(module) == torch.nn.Linear:
        module.weight.fill_(value)
    

@pytest.fixture(scope='module')
def model():
    net = torch.nn.Sequential(
        torch.nn.Linear(5, 25),
        torch.nn.ReLU(),
        torch.nn.Linear(25, 25),
        torch.nn.Linear(25, 2)
    )
    # intialize with magic value
    magic = 1701.0
    _init = partial(init, value=magic)
    with torch.no_grad():
        net.apply(_init)
    return net


def method_factory(classname: str, method_name: str) -> callable:

    def _method(*args, **kwargs):
        print(f'{classname} {method_name} request with args {args} and kwargs {kwargs}')
    
    return _method


class MockCRDHandler:
    """Instead of real IO, print create, read and delete method requests to stdout."""
    def __init__(self, directory=None):
        self.directory = directory

        # create mock methods that report invocation to stdout
        methods = ('read', 'write', 'delete')
        for method in methods:
            setattr(self, method, method_factory(self.__class__.__name__, method))



def test_register_on_empty_registry(model, tmp_path):
    capacity = 4
    pref = ScorePreference.HIGHER_IS_BETTER
    handler = RWDHandler(directory=tmp_path)

    registry = Registry(capacity=capacity, score_preference=pref,
                        rwd_handler=handler)

    score = 0.9
    registry.register(item=(score, model))
    assert registry.population == 1
    assert np.allclose(registry.scores, [score])


@pytest.mark.parametrize(('preference', 'reverse'),
                         [(ScorePreference.HIGHER_IS_BETTER, True),
                          (ScorePreference.LOWER_IS_BETTER, False)])
def test_registry_sorts_best_to_worst(model,
                                      preference,
                                      reverse):
    capacity = 6
    handler = MockCRDHandler()
    registry = Registry(capacity=capacity, score_preference=preference,
                        rwd_handler=handler)

    all_scores = [0.6, 0.8, 0.3, 0.5, 0.33, 0.75]
    # prefill two items to allow ordering
    for s in all_scores[:2]:
        registry.register(item=(s, model))

    # fill registry
    for i, s in enumerate(all_scores[2:], start=2):
        retitem = registry.register(item=(s, model))
        expected_scores = np.array(sorted(all_scores[:i+1], reverse=reverse))
        registry_scores = np.array(registry.scores)
        assert np.allclose(registry_scores, expected_scores)


def test_registry_population_does_not_exceed_capacity(model, tmp_path):
    capacity = 3
    pref = ScorePreference.HIGHER_IS_BETTER
    handler = RWDHandler(directory=tmp_path)

    registry = Registry(capacity=capacity, score_preference=pref,
                        rwd_handler=handler)

    # fill registry
    for s in [0.4, 0.5, 0.6]:
        registry.register(item=(s, model))
        assert registry.population <= capacity
    # add one more item that should expel one wasteitem
    wasteitem = registry.register(item=(0.9, model))
    assert registry.population <= capacity
    assert np.isclose(wasteitem.score, 0.4)
