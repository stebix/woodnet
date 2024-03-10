# save top k instances of the model 
import logging
import torch
import importlib

from enum import Enum
from functools import cached_property
from numbers import Number
from datetime import datetime
from pathlib import Path

from woodnet.checkpoint import ScoredCheckpoint
from woodnet.checkpoint.handlers import RWDHandler


LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


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



class Registry:
    """
    This fixed-size registry keeps track of model instances and their corresponding
    evaluation score. A user-set number of optimal instances are stored. 
    """
    def __init__(self,
                 capacity: int,
                 score_preference: ScorePreference,
                 rwd_handler: RWDHandler) -> None:
        
        self._capacity: int = capacity
        self._score_preference: ScorePreference = score_preference
        self._content: list[ScoredCheckpoint] = []
        self.rwd_handler: RWDHandler = rwd_handler
            
    @property
    def population(self) -> int:
        """Number of actual registry entries."""
        return len(self._content)
    
    @property
    def capacity(self) -> int:
        """Maximum number of registry entries."""
        return self._capacity
    
    @property
    def score_preference(self) -> ScorePreference:
        """Score preference."""
        return self._score_preference
    
    @cached_property
    def score_optimality_function(self) -> callable:
        """Return the optimality function depending on the score preference."""
        if self._score_preference is ScorePreference.HIGHER_IS_BETTER:
            return max
        else:
            return min

    @property
    def scores(self) -> list[Number]:
        return [chkpt.score for chkpt in self._content]

    @cached_property
    def initial_score(self) -> Number:
        return INITIAL_SCORES[self._score_preference]

    @property
    def current_optimal_score(self) -> Number:
        """Return the current optimal score of all registry elements."""
        if self.population == 0:
            return self.initial_score
        return self.score_optimality_function(self.scores)

    @property
    def current_max_score(self) -> Number:
        if self.population == 0:
            return self.initial_score
        return max(self.scores)

    @property
    def current_min_score(self) -> Number:
        if self.population == 0:
            return self.initial_score
        return min(self.scores)

    def register(self, item: tuple[Number, torch.nn.Module]) -> None | ScoredCheckpoint:
        """Insert a new item into the registry.
        
        Returns either
            - `None` : if the insertion could be performed without excising a
                       preexisting item. This is usually the case when the population is
                       lower than the capacity of the registry.
                       
            - `tuple` : if the insertion triggered the excision of an item.
                        This is the case when the score of the inserted item is better
                        than the worst-scoring item inside the registry.
        """
        score, model = item
        rank = self.query_score_rank(score)

        if rank in {ScoreRank.OPTIMAL, ScoreRank.FEASIBLE}:
            qualifier = 'optimal' if rank is ScoreRank.OPTIMAL else None
            # TODO: maybe move savepath generation back into CRDHandler definition
            savepath = self.rwd_handler.write(model=model, qualifier=qualifier)
            chkpt = ScoredCheckpoint(filepath=savepath, score=score)

            self._content.append(chkpt)
            # always keep registry content sorted best-to-worst score
            reverse = True if self._score_preference is ScorePreference.HIGHER_IS_BETTER else False
            self._content = sorted(self._content, key=lambda c: c.score, reverse=reverse)

            logger.debug(f'registered new {qualifier or "feasible"} registry item <{score}>')

            if len(self._content) > self.capacity:
                # pop worst-scoring item from the content list
                wasteitem = self._content.pop(-1)
            else:
                wasteitem = None
            
            return wasteitem

        # score is not feasible - we do not need to do anythiong further
        logger.debug(f'no-op: attempted registration of checkpoint with futile score < {score} >')
        return None

        
    def query_score_rank(self, score: Number) -> ScoreRank:
        """
        Checks if the provided score constitutes a new optimal score, a feasible improvement
        or a non-viable, futile score compared to preexisting registry members.
        """
        if self.population < self.capacity:
            return ScoreRank.FEASIBLE

        if self._score_preference is ScorePreference.HIGHER_IS_BETTER:

            if score > self.current_max_score:
                return ScoreRank.OPTIMAL
            elif score > self.current_min_score:
                return ScoreRank.FEASIBLE
            else:
                return ScoreRank.FUTILE

        elif self._score_preference is ScorePreference.LOWER_IS_BETTER:

            if score < self.current_min_score:
                return ScoreRank.OPTIMAL
            elif score < self.current_max_score:
                return ScoreRank.FEASIBLE
            else:
                return ScoreRank.FUTILE

        else:
            raise RuntimeError('hic sunt dracones')
            
            
    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}(capacity={self._capacity}, '
        s += f'score_preference={self._score_preference}, population={self.population})'
        return s
    
    
    def __str__(self) -> str:
        return repr(self)
    

    def emit_scoresheet(self) -> dict:
        timestamp: str = datetime.now().isoformat(timespec='milliseconds')
        data = {
            'timestamp' : timestamp,
            'capacity' : self.capacity,
            'score_preference' : self._score_preference.value,
            'scores' : {
                str(chkpt.filepath) : chkpt.score for chkpt in self._content
            }
        }
        return data


def create_default_checkpoint_directory(basedir: str | Path) -> Path:
    checkpoint_dir = Path(basedir) / 'checkpoints'
    if checkpoint_dir.exists():
        if checkpoint_dir.is_dir():
            logger.info(f'Registry using preexisting checkpoint directory: \'{checkpoint_dir}\'')
        else:
            raise FileExistsError(f'Registry initialization failed with pre-existing '
                                  f'file or object at desired checkpoint directory: \'{checkpoint_dir}\'')
    else:
        logger.info(f'Registry uses tentative checkpoint directory with path: \'{checkpoint_dir}\'')
    return checkpoint_dir



def get_registry_class(classname: str) -> type[Registry]:
    """
    Programmatically retrieve the registry classes by a string name.
    """
    module_name = 'woodnet.checkpoint.registry'
    module = importlib.import_module(name=module_name)
    try:
        class_ = getattr(module, classname)
    except KeyError:
        raise ValueError(
            f'Could not retrieve class \'{classname}\' from module \'{module}\''
        )
    return class_



def create_score_registry(configuration: dict,
                          checkpoint_directory: str | Path | None) -> Registry:
    """
    Create the score registry from the top-level configuration dictionary.
    """
    registry_config = configuration['trainer']['score_registry']
    capacity = registry_config.get('capacity', 1)
    preference = ScorePreference(registry_config['score_preference'])
    registry_class = get_registry_class(registry_config['name'])

    if not checkpoint_directory:
        logger.info('Initializing registry checkpoint directory via deduction '
                    'from top-level configuration.')
        checkpoint_directory = create_default_checkpoint_directory(
            configuration['experiment_directory']
        )
    else:
        checkpoint_directory = Path(checkpoint_directory)
    
    handler = RWDHandler(directory=checkpoint_directory)
    registry = registry_class(capacity=capacity, score_preference=preference,
                              rwd_handler=handler)
    return registry








