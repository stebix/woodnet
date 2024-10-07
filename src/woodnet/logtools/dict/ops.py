"""
Functional tools for logged item retrieval from mapping objects,
"""
import logging
from typing import Hashable, Literal, Any
from collections.abc import Mapping, MutableMapping

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


def retrieve(d: MutableMapping, /, key: Hashable, default: Any,
             method: Literal['get', 'pop'],
             prefix: str = '', suffix: str = '') -> Any:
    """
    Retrieve a value of mutable mapping and log whether actual or default value was retrieved.

    Parameters
    ----------

    d : MutableMapping
        Base container.

    key : Hashable
        Key for the desired value.

    default : Any
        Default value returned if key - value pair is not present.

    method : Literal
        Choose between value get (non-modifying) and value pop
        (in-place modification).

    prefix : str, optional
        Set additional prefix information for the key.
        Defaults to empty string. 

    suffix : str, optional
        Set additional suffix information for the key.
        Defaults to empty string. 

    Returns
    -------

    value : Any
        Either the value defined by the dictionary or the default value.
    """
    sentinel = object()
    retrieve = d.get if method == 'get' else d.pop
    value = retrieve(key, sentinel)
    expanded_key = ''.join((prefix, key, suffix))
    if value is sentinel:
        value = default
        logger.info(f'Using \'{expanded_key}\' with internal default value < {value} >')
    else:
        logger.info(f'Using \'{expanded_key}\' with configuration value < {value} >')
    return value


def get(d: Mapping, /, key: Hashable, default: Any,
        prefix: str = '', suffix: str = '') -> Any:
    """
    Get (key, value) pair of mapping and emit log message.

    Parameters
    ----------

    d : Mapping

    key : Hashable

    default : Any
        Returned default value if `key` does not exist in `d`.
    
    prefix : str, optional
        Prefix string prepended to `key` inside the log message.
        Defaults to empty string.

    suffix : str, optional
        Suffix string appended to `key` inside the log message.
        Defaults to empty string.
    """
    return retrieve(d, key=key, default=default, method='get',
                    prefix=prefix, suffix=suffix)


def pop(d: MutableMapping, /, key: Hashable, default: Any,
        prefix: str = '', suffix: str = '') -> Any:
    """
    Pop (key, value) pair of mutable mapping and emit log message.

    Parameters
    ----------

    d : Mapping

    key : Hashable

    default : Any
        Returned default value if `key` does not exist in `d`.
    
    prefix : str, optional
        Prefix string prepended to `key` inside the log message.
        Defaults to empty string.

    suffix : str, optional
        Suffix string appended to `key` inside the log message.
        Defaults to empty string.
    """
    return retrieve(d, key=key, default=default, method='pop',
                    prefix=prefix, suffix=suffix)