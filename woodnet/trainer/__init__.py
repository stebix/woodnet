import importlib
from typing import Type, Union

from woodnet.trainer.base_trainer import Trainer
from woodnet.trainer.legacy_trainer import LegacyTrainer


TrainerClass = Union[Type[Trainer], Type[LegacyTrainer]]


def retrieve_trainer_class(name: str) -> TrainerClass:
    """Retrieve any trainer class by its string name."""
    module_names = [
        'woodnet.trainer.base_trainer',
        'woodnet.trainer.legacy_trainer'
    ]
    for module_name in module_names:
        module = importlib.import_module(module_name)
        try:
            return getattr(module, name)
        except AttributeError:
            pass
    
    raise ValueError(f'unrecognized trainer class name \'{name}\'')
