import pytest
from typing import Sequence

from woodnet.logtools.dict.dict import RecursiveLoggedDict


def log_method_factory(cache_name: str):
    """Wow, such Python dynamism :)"""
    def log(self, message: str, **kwargs):
        cache = getattr(self, cache_name)
        cache.append((message, kwargs))

    return log



class MockLogger:
    """
    MockLogger object emulating the standard library logger object.

    Features:
        - dynamically support levels (modify them via the levels class variable)
        - 
    """
    levels: list[str] = ['debug', 'info', 'warning', 'error', 'critical']

    def __init__(self) -> None:
        self.add_methods(self.levels)
        self._create_caches(self.levels)
    
    def _create_caches(self, levels: Sequence[str]) -> None:
        for level in levels:
            setattr(self, self._make_cache_attribute_name(level), [])

    @staticmethod
    def _make_cache_attribute_name(level: str) -> str:
        return f'{level}_message_cache'

    @classmethod
    def add_methods(cls, levels: Sequence[str]) -> None:
        """Add the MockLogger logging methods that directly derive from the levels."""
        for level in levels:
            cache_name = cls._make_cache_attribute_name(level)
            method = log_method_factory(cache_name)
            setattr(cls, level, method)
        



def test_initialization_from_dict():
    num = 1701
    testdata = {
        'captain' : 'jean-luc picard',
        'registration_number' : num,
        'sub_char' : 'D',
        'shuttles' : {
            'magellan' : 'available',
            'el-baz' : 'repairing',
            'goddard' : 'destroyed'
        }
    }
    d = RecursiveLoggedDict(testdata)
    assert d['registration_number'] == num


def test_get_emits_log_message():
    testdata = {
        'captain' : 'jean-luc picard',
        'registration_number' : 1701,
        'sub_char' : 'D'
    }
    logger = MockLogger()
    d = RecursiveLoggedDict(testdata, logger)
    key = 'captain'
    value = d.get(key)
    # access the first and only log message
    message, _ = logger.info_message_cache[0]
    expected_message = f'Using \'{key}\' with configuration value < {value} >'
    assert message == expected_message



def test_pop_actually_removes_key_value_pair():
    testdata = {
        'captain' : 'jean-luc picard',
        'registration_number' : 1701,
        'sub_char' : 'D',
        'shuttles' : {
            'magellan' : 'available',
            'el-baz' : 'repairing',
            'goddard' : 'destroyed'
        }
    }
    logger = MockLogger()
    d = RecursiveLoggedDict(testdata, logger)
    key = 'sub_char'
    value = d.pop(key)
    # check value result and dictionary mutation
    assert value == 'D'
    assert key not in d.keys()
    assert value not in d.values()



def test_returns_RecursiveLoggedDict_instance_upon_retrieval():
    testdata = {
        'captain' : 'jean-luc picard',
        'registration_number' : 1701,
        'sub_char' : 'D',
        'shuttles' : {
            'magellan' : 'available',
            'el-baz' : 'repairing',
            'goddard' : 'destroyed'
        }
    }
    d = RecursiveLoggedDict(testdata)
    
    subd = d.get('shuttles')
    assert isinstance(subd, RecursiveLoggedDict), f'expected {RecursiveLoggedDict}, got {type(subd)}'


def test_recursive_cast_upon_retrieval():
    """Test that multiply nested Mappings are cast upon retrieval."""
    testdata = {
        'shuttles' : {
            'magellan' : 'available',
            'el-baz' : 'repairing',
            'goddard' : {
                'weapons' : {
                    'quantum_torpedoes' : {'launcher_1' : 'full'},
                    'phasers' : {'bank_1' : 'pristine'}
                }
            }
        }
    }
    testdata = RecursiveLoggedDict(testdata)
    keys = ['shuttles', 'goddard', 'weapons', 'phasers']
    for key in keys:
        retval = testdata.get(key)
        assert isinstance(retval, RecursiveLoggedDict)
        testdata = retval