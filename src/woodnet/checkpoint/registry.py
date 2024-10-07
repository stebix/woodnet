# save top k instances of the model 
import logging
import torch
import importlib

from collections.abc import Mapping
from functools import cached_property
from numbers import Number
from datetime import datetime
from pathlib import Path

from woodnet.checkpoint import ScoredCheckpoint
from woodnet.checkpoint.scores import ScorePreference, ScoreRank, INITIAL_SCORES
from woodnet.checkpoint.handlers import RWDHandler
from woodnet.logtools.dict import LoggedDict

LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(LOGGER_NAME)


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
        if score_preference is ScorePreference.HIGHER_IS_BETTER:
            self._reverse: bool = True
        else:
            self._reverse: bool = False
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

        if rank is ScoreRank.FUTILE:
            # score is not feasible - we do not need to do anything further
            logger.info(f'no-op: attempted registration of checkpoint with futile score < {score} >')
            return None

        qualifier = 'optimal' if rank is ScoreRank.OPTIMAL else None
        # TODO: maybe move savepath generation back into CRDHandler definition
        savepath = self.rwd_handler.write(model=model, qualifier=qualifier)
        chkpt = ScoredCheckpoint(filepath=savepath, score=score, rank=rank)

        # demote the pre-existing optimal ScoredCheckpoint instance 
        if rank is ScoreRank.OPTIMAL:
            self.demote_current_optimal()

        self._content.append(chkpt)
        # always keep registry content sorted optimal-to-pessimal score
        self._content = sorted(self._content, key=lambda c: c.score, reverse=self._reverse)

        logger.debug(f'registered new {qualifier or "feasible"} registry '
                     f'item with metric score < {score} >')

        if len(self._content) > self.capacity:
            # pop worst-scoring item from the content list
            wasteitem = self._content.pop(-1)
        else:
            wasteitem = None
        
        return wasteitem


    def demote_current_optimal(self) -> None:
        """Demote the current optimal ScoredCheckpoint member to `ScoreRank.FEASIBLE`"""
        for checkpoint in self._content:
            if checkpoint.rank is ScoreRank.OPTIMAL:
                checkpoint.demote()
                return

        
    def query_score_rank(self, score: Number) -> ScoreRank:
        """
        Checks if the provided score constitutes a new optimal score, a feasible improvement
        or a non-viable, futile score compared to preexisting registry members.
        """
        if self.population < self.capacity:
            # here we can always keep the insertion, but we need to know optimality
            if (self._score_preference is ScorePreference.HIGHER_IS_BETTER and
                    score > self.current_max_score):
                return ScoreRank.OPTIMAL
            if (self._score_preference is ScorePreference.LOWER_IS_BETTER and
                    score < self.current_min_score):
                return ScoreRank.OPTIMAL
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




DEFAULT_REGISTRY_CLASS_NAME: str = 'Registry'
DEFAULT_CAPACITY: int = 1
DEFAULT_SCORE_PREFERENCE_STR: str = 'higher_is_better'


def create_score_registry(configuration: Mapping | None,
                          checkpoint_directory: str | Path) -> Registry:
    """
    Create the score registry from configuration dictionary. If not given,
    create registry in its default flavour.

    Parameters
    ----------

    configuration :  Mapping or None
        Configuration for the score registry. Default values will be used
        if configuration is `None`.

    checkpoint_directory : str or pathlib.Path
        Directory where the registry will save the checkpoints.
        Must exist.

    Returns
    -------

    registry : Registry
        Initialized registry object.
    """
    checkpoint_directory = Path(checkpoint_directory)
    if not checkpoint_directory.is_dir():
        raise NotADirectoryError('creation of registry requires preexisting checkpoint directory')
    configuration = LoggedDict(configuration or {}, logger)
    registry_class = get_registry_class(configuration.get('name', DEFAULT_REGISTRY_CLASS_NAME))
    capacity = configuration.get('capacity', default=DEFAULT_CAPACITY)
    preference = ScorePreference(
        configuration.get('score_preference', default=DEFAULT_SCORE_PREFERENCE_STR)
    )
    handler = RWDHandler(checkpoint_directory)
    return registry_class(capacity=capacity, score_preference=preference, rwd_handler=handler)
