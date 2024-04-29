"""
Encapsulate tooling for loading and preprocessing the
evaluation result files as pandas dataframes.

@jsteb 2024
"""
import numpy as np
import pandas as pd

from collections.abc import Mapping, Sequence


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
    expanded_df = pd.concat([df, additional_dfs])
    expanded_df.sort_index()
    return expanded_df
    