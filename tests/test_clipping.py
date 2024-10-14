import torch
import pytest

from woodnet.gradtools.clipping import create_gradclip_func
from woodnet.custom.exceptions import ConfigurationError



def test_with_valid_norm_clipping_configuration():
    conf = {
        'name' : 'grad_norm',
        'max_norm' : 1.0,
        'norm_type' : 2.0
    }
    func = create_gradclip_func(conf)
    assert func is not None
    


def test_with_valid_value_clipping_configuration():
    conf = {
        'name' : 'grad_value',
        'clip_value' : 1.0,
        'foreach' : None
    }
    func = create_gradclip_func(conf)
    assert func is not None


def test_fail_with_missing_name():
    conf = {
        'max_norm' : 1.0,
        'norm_type' : 2.0
    }
    with pytest.raises(ConfigurationError):
        create_gradclip_func(conf)


def test_fail_with_invalid_name():
    conf = {
        'name' : 'invalid_name',
        'max_norm' : 1.0,
        'norm_type' : 2.0
    }
    with pytest.raises(ConfigurationError):
        create_gradclip_func(conf)

