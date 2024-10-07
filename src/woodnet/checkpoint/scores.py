"""
Score enums.

@jsteb 2024
"""
from enum import Enum


class ScorePreference(Enum):
    HIGHER_IS_BETTER = 'higher_is_better'
    LOWER_IS_BETTER = 'lower_is_better'


class ScoreRank(Enum):
    OPTIMAL = 'optimal'
    FEASIBLE = 'feasible'
    FUTILE = 'futile'


INITIAL_SCORES: dict[ScorePreference, float] = {
    ScorePreference.HIGHER_IS_BETTER :  float('-inf'),
    ScorePreference.LOWER_IS_BETTER :  float('+inf'),
}
