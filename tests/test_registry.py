import pytest
import torch
import numpy as np
import math

from functools import partial

from woodnet.checkpoint.handlers import RWDHandler
from woodnet.checkpoint.registry import *


def is_posinf(value):
    return math.isinf(value) and value > 0

def is_neginf(value):
    return math.isinf(value) and value < 0


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
    # initialize with magic value
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
                                      reverse,
                                      tmp_path):
    capacity = 6
    handler = RWDHandler(directory=tmp_path)
    registry = Registry(capacity=capacity, score_preference=preference,
                        rwd_handler=handler)

    pre_scores = [0.6, 0.8, 0.3, 0.5]
    post_scores = [0.33, 0.75]
    # prefill two items to allow ordering
    for s in pre_scores:
        registry.register(item=(s, model))

    # fill registry
    for i, s in enumerate(post_scores, start=1):
        retitem = registry.register(item=(s, model))
        scores = pre_scores + post_scores[:i]
        expected_scores = np.array(sorted(scores, reverse=reverse))
        registry_scores = np.array(registry.scores)
        assert np.allclose(registry_scores, expected_scores)
        assert retitem is None, 'capacity should allow all items inside'


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


@pytest.mark.parametrize(('preference', 'infcheck_func'),
                         [(ScorePreference.HIGHER_IS_BETTER, is_neginf),
                          (ScorePreference.LOWER_IS_BETTER, is_posinf)])
def test_returns_initial_scores_on_empty_registry(tmp_path, preference, infcheck_func):
    handler = RWDHandler(directory=tmp_path)
    registry = Registry(capacity=3, score_preference=preference,
                        rwd_handler=handler)
    assert infcheck_func(registry.current_min_score)
    assert infcheck_func(registry.current_max_score)


def test_emit_scoresheet(tmp_path, model):
    cap = 3
    handler = RWDHandler(directory=tmp_path)
    preference = ScorePreference.HIGHER_IS_BETTER
    registry = Registry(capacity=cap, score_preference=preference,
                        rwd_handler=handler)
    
    # add fake checkpoint data
    preset_scores = (0.4, 0.3, 0.75, 0.95)
    for score in preset_scores:
        registry.register(item=(score, model))

    scoresheet = registry.emit_scoresheet()
    
    assert isinstance(scoresheet, dict)
    assert scoresheet['capacity'] == cap

    # get scores from the scoresheet and compare the sorted sequences
    scores = tuple(
        sorted([s for s in scoresheet['scores'].values()])
    )
    expected_scores = tuple(sorted(preset_scores))[1:]
    assert scores == expected_scores


def test_get_registry_class():
    classname = 'Registry'
    expected_class = Registry
    class_ = get_registry_class(classname=classname)    
    assert class_ == expected_class


def test_create_registry_from_valid_configuration(tmp_path):
    # basal configuration settings
    expected_capacity = 3
    expected_preference_str = 'higher_is_better'
    expected_preference = ScorePreference(expected_preference_str)
    
    checkpoint_directory = tmp_path / 'checkpoints'
    checkpoint_directory.mkdir()
    assert checkpoint_directory.is_dir(), f'test setup failure: not dir @ \'{checkpoint_directory}\''

    config = {
        'name' : 'Registry',
        'capacity' : expected_capacity,
        'score_preference' : expected_preference_str
    }
    registry = create_score_registry(config,
                                     checkpoint_directory=checkpoint_directory)
    
    # test some properties of the registry
    assert registry.capacity == expected_capacity
    assert registry.score_preference == expected_preference
    assert registry.rwd_handler.directory == checkpoint_directory



