from typing import Type, Union

from woodnet.trainer.base_trainer import Trainer
from woodnet.trainer.legacy_trainer import LegacyTrainer


TrainerClass = Union[Type[Trainer], Type[LegacyTrainer]]


def retrieve_trainer_class(name: str) -> TrainerClass:
    """Retrieve any trainer class by its string name."""
    modules = [
        'woodnet.trainer.base_trainer',
        'woodnet.trainer.legacy_trainer'
    ]
    for module in modules:
        try:
            return getattr(module, name)
        except AttributeError:
            pass
    
    raise ValueError(f'unrecognized trainer class name \'{name}\'')
