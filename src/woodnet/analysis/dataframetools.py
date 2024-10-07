"""
Encapsulate tooling for loading and preprocessing the
evaluation result files as pandas dataframes.

@jsteb 2024
"""
import pickle
import numpy as np
import pandas as pd

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from frozendict import frozendict

############### Tooling basic loading and wrangling the nested dictionary 

INDEX_NAMES: list[str] = ['fold', 'model_ID', 'transform', 'stage', 'metric']

# The loaded structure is a deeply nested dict. These type hints should
# clarify the level we are operating at.
TransformResultMapping = Mapping
InstanceResultMapping = Mapping
MultiInstanceResultMapping = Mapping
FoldwiseResultMapping = Mapping


def construct_stage_mapping(result: TransformResultMapping) -> dict[str, frozendict]:
    stage_mapping: dict[str, frozendict] = {}
    for i, parameters in enumerate(result['results'].keys(), start=1):
        stage_mapping[f'stage_{i}'] = parameters
    return stage_mapping


def invert(d: Mapping) -> dict:
    return {v : k for k, v in d.items()}


def remodel_transform_result(result: TransformResultMapping) -> tuple[dict, dict]:
    """
    Convert the parameterization information contained in the `frozendict` keys
    into a generic `stage-$N` ID string. Returns a copy.
    """
    # extract necessary information
    metadata = result.get('metadata')
    parametrizations = result.get('results')
    # setup conversion and cleaning
    stage_mapping = construct_stage_mapping(result)
    fdict_to_stage = invert(stage_mapping)
    converted = {}
    for old_key, value in parametrizations.items():
        new_key = fdict_to_stage.get(old_key)
        converted[new_key] = deepcopy(value)
    return (converted, stage_mapping)


def remodel_instance_result(result: InstanceResultMapping) -> tuple[dict, dict]:
    remodeled_transform_results = {}
    transform_stage_mappings = {}
    for transform_name, transform_result in result.items():
        remodulation, stage_mapping = remodel_transform_result(transform_result)
        remodeled_transform_results[transform_name] = remodulation
        transform_stage_mappings[transform_name] = stage_mapping
    return (remodeled_transform_results, transform_stage_mappings)


def remodel_multiinstance_result(result: MultiInstanceResultMapping) -> tuple[dict, dict]:
    remodeled_instance_results = {}
    instance_wise_stage_mappings = {}
    for mdl_instance_ID, instance_result in result.items():
        remodulation, stage_mappings = remodel_instance_result(instance_result)
        remodeled_instance_results[mdl_instance_ID] = remodulation
        instance_wise_stage_mappings[mdl_instance_ID] = stage_mappings
    return (remodeled_instance_results, instance_wise_stage_mappings)

    
def remodel_foldwise_result(result: FoldwiseResultMapping) -> tuple[dict, dict]:
    remodeled_multiinstance_results = {}
    multiinstance_wise_stage_mappings = {}
    for fold_ID, multiinstance_result in result.items():
        remodulation, stage_mappings = remodel_multiinstance_result(multiinstance_result)
        remodeled_multiinstance_results[fold_ID] = remodulation
        multiinstance_wise_stage_mappings[fold_ID] = stage_mappings
    return (remodeled_multiinstance_results, multiinstance_wise_stage_mappings)
    

def as_tuplekeyed(dictionary: dict) -> list:
    """
    Transform a nested dictionary into a list of 2-tuples.
    The first element is the 'trace' of keys traversed to arrive
    at the value, with the discovered value as the second element.
    """
    cache = []
    
    def _into(dictionary, prefix=()):
        for key, value in dictionary.items():
            current_prefix = (*prefix, key)
            if isinstance(value, dict):
                _into(value, current_prefix)
                continue 
            cache.append((current_prefix, value))
    
    _into(dictionary)        
    return cache


def to_dataframe(tuplekeyed: list[tuple]) -> pd.DataFrame:
    """
    Transform tuplekeyed data into a pandas DataFrame
    with a hierarchical multiindex.
    """
    indices, values = zip(*tuplekeyed)
    df = pd.DataFrame(values, index=pd.MultiIndex.from_tuples(indices))
    return df
    


def preprocess_evaluation_result(result: Mapping) -> pd.DataFrame:
    """Integrated preprocessing of a result mapping into a pandas data frame."""
    result, _ = remodel_foldwise_result(result)
    # TODO: Maybe insert check for congruency here
    tuplekeyed = as_tuplekeyed(result)
    df = to_dataframe(tuplekeyed)
    df = df.rename_axis(index=tuple(INDEX_NAMES))
    return df


def sort_at_stagelevel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the multi-indexed results dataframe at the 'stage' level
    of the multi-index. Useful after the additioan of the identity
    transform as stage 0, since then ordering of the stages is
    not ascending. 
    """
    stagelevel = df.index.names.index('stage')
    return df.sort_index(level=stagelevel, inplace=False, ascending=True)


############### Tooling for transforming a loaded and properly indexed dataframe
# (i.e. multiindex with correct index level names) by adding the null/identity
# as the 'stage-0' element to every individual transformation

NameMapping = Mapping[str, str]
LevelMapping = Mapping[int, str]


def get_uniques(df: pd.DataFrame, level: int) -> np.ndarray:
    """Get the unique values of a multi-indexed dataframe at the given index level."""
    values = df.index.get_level_values(level)
    return np.unique(values)


def all_isinstance(s: Sequence, class_or_tuple) -> bool:
    return all(isinstance(elem, class_or_tuple) for elem in s)


def is_name_mapping(m: Mapping) -> bool:
    all_str_keys = all_isinstance(m.keys(), str)
    all_str_vals = all_isinstance(m.values(), str)
    return all_str_keys and all_str_vals


def is_level_mapping(m: Mapping) -> bool:
    all_int_keys = all_isinstance(m.keys(), int)
    all_str_vals = all_isinstance(m.values(), str)
    return all_int_keys and all_str_vals


def _legacy_transform_index(index: pd.MultiIndex, value: str, depth: int) -> pd.MultiIndex:
    index_depth = len(index[0])
    if depth < 0 or depth >= index_depth:
        raise ValueError(f'depth for the index transformation must be between '
                         f'0 and index depth ({index_depth}), but got depth = {depth}')
    tuples = []
    for idx_elem in index:
        parts = [
            value if i == depth else part
            for i, part in enumerate(idx_elem)
        ]
        tuples.append(tuple(parts))
    return pd.MultiIndex.from_tuples(tuples, names=index.names)


def _level_mapping_transform_index(index: pd.MultiIndex, mapping: Mapping[int, str]) -> pd.MultiIndex:
    """
    Transform a pandas multiindex via a mapping.
    The mapping must specify the mapping from the level of the index (integer key)
    to the new values of the new index at that level (string value).
    
    Parameters
    ----------
    
    index : pd.MultiIndex
        Basic template index on which the transform is based on. Non-transformed
        elements and names will be copied over.
        
    mapping : Mapping[int, str]
        Remapping of the index depth positions to the new values
        
    Returns
    -------
    
    new_index : pd.MultiIndex
        Newly created index with transformed elements.
    """
    index_depth = len(index[0])
    if any(k < 0 for k in mapping.keys()) or any(k >= index_depth for k in mapping.keys()):
        raise ValueError(f'depth keys for the index transformation must be between '
                         f'0 and index depth {index_depth}, but got depths = {mapping.keys()}')
    tuples = []
    for idx_elem in index:
        parts = [
            mapping.get(i, part)
            for i, part in enumerate(idx_elem)
        ]
        tuples.append(tuple(parts))
    return pd.MultiIndex.from_tuples(tuples, names=index.names)



def _name_mapping_transform_index(index: pd.MultiIndex, mapping: Mapping[str, str]) -> pd.MultiIndex:
    """
    Transform a pandas multiindex via a mapping.
    The mapping must specify the mapping from the level name of the index (string key)
    to the new values of the new index at that level (string value).
    
    Parameters
    ----------
    
    index : pd.MultiIndex
        Basic template index on which the transform is based on. Non-transformed
        elements and names will be copied over.
        
    mapping : Mapping[str, str]
        Remapping of the index level names to the new values
        
    Returns
    -------
    
    new_index : pd.MultiIndex
        Newly created index with transformed elements.
    """
    index_names = set(index.names)
    remap_names = set(mapping.keys())
    is_subset = remap_names <= index_names
    if not is_subset:
        diff = remap_names - index_names
        raise ValueError(
            f'Mismatch of current and intended remapped index names. Set difference: {diff}'
        )
    # transform the name-wise mapping into a level-wise mapping
    level_mapping = {}
    for name, value in mapping.items():
        level = index.names.index(name)
        level_mapping[level] = value
    
    return _level_mapping_transform_index(index, level_mapping)




def mapping_transform_index(index: pd.MultiIndex,
                            mappings: Sequence[NameMapping] | Sequence[LevelMapping]
                            ) -> list[pd.MultiIndex]:
    """
    Perform an arbitrary number of remapping transformations on a `pd.MultiIndex`,
    meaning that this is a one-to-many transform.
    """
    if all(is_name_mapping(mapping) for mapping in mappings):
        transform_func = _name_mapping_transform_index
    elif all(is_level_mapping(mapping) for mapping in mappings):
        transform_func = _level_mapping_transform_index
    else:
        raise ValueError('mappings do not uniformly conform to either \'NameMapping\' '
                         'type or \'LevelMapping\' type')
    indices: list[pd.MultiIndex] = []
    for mapping in mappings:
        indices.append(transform_func(index, mapping))
    return indices


def remodel_dataframe_by_remapping(df: pd.DataFrame,
                                   remappings: Sequence[NameMapping] | Sequence[LevelMapping]
                                   ) -> list[pd.DataFrame]:
    """
    Make (multiple) remodelings of the basal input dataframe by modification
    of the `pd.MultiIndex`-typed index.
    """
    remodels = []
    indices = mapping_transform_index(df.index, remappings)
    for index in indices:
        # we require a copy since we assign to the index: this is one of the
        #  few in-place mutating pandas actions
        dfc = df.copy()
        dfc.index = index
        remodels.append(dfc)
    return remodels


def add_identity_as_stage_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform dataframe by adding the identity metrics results as the stage-0
    element for all transformations.
    """
    # first get a subselection of the results for the 'Identity' transform for
    # all other indices.
    # Expected index name setup:
    # (fold, model_ID, transform, stage, metric)
    id_subsec = df.loc[(slice(None), slice(None), 'Identity', slice(None), slice(None)), :]
    transform_level: int = df.index.names.index('transform')
    transforms: np.ndarray = get_uniques(df, level=transform_level) # should be of type 'object'
    # Exclude identity since that is already present. We only wnat to insert the
    # identity metrics values for every transformation as 'stage-0' element
    transforms = transforms[transforms != 'Identity']
    remappings = [
        {'transform' : transform, 'stage' : 'stage-0'}
        for transform in transforms
    ]
    additional_dfs = remodel_dataframe_by_remapping(id_subsec, remappings)
    expanded_df = pd.concat([df, *additional_dfs])
    expanded_df.sort_index()
    return expanded_df




def load_from_pkl(filepath: Path | str) -> pd.DataFrame:
    """
    Fully integrated loading and processing of results data.
    Automatically converts to multi-indexed DataFrame and adds the identity
    transformation as stage-0 for all other transforms.
    """
    filepath = Path(filepath) if not isinstance(filepath, Path) else filepath

    with filepath.open(mode='rb') as handle:
        data = pickle.load(handle)
    
    data = data['folds']
    df = preprocess_evaluation_result(data)
    df = add_identity_as_stage_zero(df)
    df = sort_at_stagelevel(df)
    return df