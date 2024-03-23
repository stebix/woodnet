"""
Dictionary subclass with logged item retrieval..

@jsteb 2024
"""
import logging
from collections import UserDict
from collections.abc import Mapping
from typing import Hashable, Literal, Any

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)

NULL = object()


class LoggedDict(UserDict):
    """Dictionary that emits log messages upon get or pop method calls.
    
    Useful for deep learning configuration dictionaries. 
    
    The log messages reflect whether the requested key-value-pair is actually present
    in the dictionary or if the default value was used.
    """
    def __init__(self, dict=None, logger=None, /, **kwargs):
        self.logger = logger or logging.getLogger()
        super().__init__(dict, **kwargs)
        
    def retrieve(self, action: Literal['get', 'pop'], key: Hashable, default: Any = NULL) -> Any:
        sentinel = object()
        if action == 'get':
            retrieve_fn = self.data.get
        elif action == 'pop':
            retrieve_fn = self.data.pop
        else:
            raise ValueError(f'invalid retrieve action \'{action}\'')
                
        value = retrieve_fn(key, sentinel)
        
        if value is sentinel:
            if default is NULL:
                raise KeyError(f'could not {action} \'{key}\': key does not exist and default is not given')
            else:
                self.logger.info(f'Using \'{key}\' with default value < {default} >')
                return default
            
        self.logger.info(f'Using \'{key}\' with configuration value < {value} >')
        return value
    
    def pop(self, key: Hashable, default: Any = NULL) -> Any:
        return self.retrieve(action='pop', key=key, default=default)
    
    def get(self, key: Hashable, default: Any = NULL) -> Any:
        return self.retrieve(action='get', key=key, default=default)



class RecursiveLoggedDict(LoggedDict):
    """
    Dictionary that emits log messages upon get or pop method calls.
    In comparison to the basic LoggedDict, this subclass specially handles
    Mapping-typed values (i.e. 'subconfigurations').
    The specialized handling boils down to:
     - specialized log message for Mapping-type values
     - Mapping-type values are cast as RecursiveLoggingDict upon retrieval
    """
    def __init__(self, dict=None, logger=None, /, **kwargs):
        super().__init__(dict, logger, **kwargs)
        
    def retrieve(self, action: Literal['get', 'pop'], key: Hashable, default: Any = NULL) -> Any:
        sentinel = object()
        if action == 'get':
            retrieve_fn = self.data.get
        elif action == 'pop':
            retrieve_fn = self.data.pop
        else:
            raise ValueError(f'invalid retrieve action \'{action}\'')
                
        value = retrieve_fn(key, sentinel)
        
        if value is sentinel:
            if default is NULL:
                raise KeyError(f'could not {action} \'{key}\': key does not exist and default is not given')
            elif isinstance(default, Mapping):
                message = f'Using \'{key}\' with default mapping < {default} >'
                default = self.cast(default)
            else:
                message = f'Using \'{key}\' with default value < {default} >'

            self.logger.info(message)
            return default
        
        if isinstance(value, Mapping):
            message = f'Retrieved \'{key}\' with configuration mapping value < {value} >'
            value = self.cast(value)
        else:
            message = f'Using \'{key}\' with configuration value < {value} >'

        self.logger.info(message)
        return value
    
    def cast(self, mapping: Mapping) -> 'RecursiveLoggedDict':
        """
        Cast mapping as instance of this RecursiveLoggedDict class and
        propagate the logger object.
        """
        return self.__class__(mapping, self.logger)
    