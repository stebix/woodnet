import logging
from collections import UserDict
from typing import Hashable, Literal, Any


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
            retrieve_fn = self.data.get
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
    