# save top k instances of the model 
import typing
from typing import Protocol

from numbers import Number
from enum import Enum

class ScorePreference(Enum):
    HIGHER_IS_BETTER = 'higher_is_better'
    LOWER_IS_BETTER = 'lower_is_better'

class Model(typing.Protocol):

    def __init__(self) -> None:
        ...

    def forward(self, *args, **kwargs):
        ...


class TopKCache:

    def __init__(self, size: int) -> None:
        self.size = size

        self._cache = {}



    def maybe_add_candidate(self, model: Model, score: Number) -> None:
        pass



if __name__ == '__main__':

    size = 4

    def init_cache(size: int, score_preference: ScorePreference) -> list:
        if score_preference is ScorePreference.LOWER_IS_BETTER:
            init_value = float('+inf')
        elif score_preference is ScorePreference.HIGHER_IS_BETTER:
            init_value = float('-inf')
        else:
            raise ValueError(
                f'invalid score preference \'{score_preference}\''
            )
        return [init_value for _ in range(size)]
    
    c = init_cache(4, ScorePreference.HIGHER_IS_BETTER)
    