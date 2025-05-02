from pathlib import Path
from collections.abc import Mapping, Sequence, Hashable
from collections import defaultdict

import attrs
import torch.utils.data as torchdata

import tqdm.auto as tqdm

from woodnet.datasets import healing
from woodnet.trainer.multiloader import MultiLoader

recipes = {
    '/resampled/augmented/vlgs-interp/sf-0_075' : {
        'dataset_build_kwargs' : {
            'planestride' : 11
        },
        'healing_kwargs' : {
            'tolerance' : 5
        },
        'loader_build_kwargs' : {
            'batch_size' : 256,
            'num_workers' : 3,
            'shuffle' : True
        },
    },
    '/resampled/augmented/vlgs-interp/sf-0_085' : {
        'dataset_build_kwargs' : {
            'planestride' : 14
        },
        'healing_kwargs' : {
            'tolerance' : 7
        },
        'loader_build_kwargs' : {
            'batch_size' : 256,
            'num_workers' : 3,
            'shuffle' : True
        },
    },
    '/resampled/augmented/vlgs-interp/sf-0_17' : {
        'dataset_build_kwargs' : {
            'planestride' : 25
        },
        'healing_kwargs' : {
            'tolerance' : 20
        },
        'loader_build_kwargs' : {
            'batch_size' : 256,
            'num_workers' : 3,
            'shuffle' : True
        },
    },
    '/resampled/augmented/vlgs-interp/sf-0_33' : {
        'dataset_build_kwargs' : {
            'planestride' : 14
        },
        'healing_kwargs' : {
            'tolerance' : 7
        },
        'loader_build_kwargs' : {
            'batch_size' : 192,
            'num_workers' : 3,
            'shuffle' : True
        },
    },
}



def create_loaders_from_recipes(
    dataset_class: type,
    paths: Sequence[Path],
    recipes: Mapping[str, Mapping],
    leave_pbar: bool = False,
) -> dict[str, torchdata.DataLoader]:
    
    dataset_build_kwargs_key: str = 'dataset_build_kwargs'
    healing_kwargs_key: str = 'healing_kwargs'
    loader_build_kwargs_key: str = 'loader_build_kwargs'
    
    loaders: dict[str, torchdata.DataLoader] = {}
    wrapped_recipes_items = tqdm.tqdm(
        recipes.items(), desc='Group Build Progress', leave=leave_pbar, unit='grp'
    )
    for ID, recipe in wrapped_recipes_items:
        display_ID = ID.split('/')[-1]
        wrapped_recipes_items.set_postfix_str(f'Group: \'{display_ID}\'')
        datasets_per_ID: list[torchdata.Dataset] = []
        wrapped_paths = tqdm.tqdm(
            paths, desc='ID Build Progress', leave=False, unit='ID'
        )
        # first step is to build all datasets for the ID/internal path
        for path in wrapped_paths:
            wrapped_paths.set_postfix_str(f'ID: \'{path.stem}\'')
            dataset_build_kwargs =  {
                'path' : path,
                'internal_path' : ID,
                **recipe[dataset_build_kwargs_key]
            }
            # this can be a sinlge dataset or a list of datasets
            # depending on the pipeline of the recipe
            datasets = dataset_class.build_from_zarr(**dataset_build_kwargs)
            if isinstance(datasets, (list, tuple)):
                datasets_per_ID.extend(datasets)
            else:
                datasets_per_ID.append(datasets)

        # second step is to heal the datasets
        wrapped_recipes_items.set_postfix_str(f'Healing \'{display_ID}\'')
        healing_kwargs = recipe[healing_kwargs_key]
        healed_datasets = healing.heal_datasets_3D(
            datasets_per_ID,
            **healing_kwargs
        )

        # third step is to create the dataloader for this ID
        wrapped_recipes_items.set_postfix_str(f'Loader build: \'{display_ID}\'')
        loader_build_kwargs = recipe[loader_build_kwargs_key]
        concat_dataset = torchdata.ConcatDataset(healed_datasets)
        dataloader = torchdata.DataLoader(
            dataset=concat_dataset,
            **loader_build_kwargs
        )
        loaders[ID] = dataloader

    return loaders


def to_multiloader(
    loaders: Mapping[Hashable, torchdata.DataLoader] | Sequence[torchdata.DataLoader],
    weights: Mapping[Hashable, float] | Sequence[float] | None = None,
) -> MultiLoader:
    """
    Convert 
    """
    if weights is None:
        loaders = list(loaders.values()) if isinstance(loaders, Mapping) else loaders
        # equal weights for all loaders - weight init is done by loader
        return MultiLoader(loaders=loaders, weights=None)
    
    if isinstance(loaders, Mapping) and isinstance(weights, Mapping):
        # both are mappings, we perform key-based matching
        keys = list(loaders.keys())
        loaders_list: list[torchdata.DataLoader] = []
        weights_list = list[float] = []
        for key in keys:
            try:
                weight = weights[key]
            except KeyError:
                msg = (f'key {key} not found in weights mapping, cannot match '
                       f'weight to loader! weights: {weights.keys()}')
                raise KeyError(msg)
            loaders_list.append(loaders[key])
            weights_list.append(weight)
        return MultiLoader(loaders=loaders_list, weights=weights_list)

    elif isinstance(loaders, Sequence) and isinstance(weights, Sequence):
        # both are sequences, we perform index-based matching inside multiloader
        return MultiLoader(loaders=loaders, weights=weights)
    else:
        # not supported - raise an error
        msg = (f'mismatch of container type for loaders and wights: '
               f'loaders: {type(loaders)}, weights: {type(weights)} '
               f'only both arguments must have either mapping '
               f'or sequence type are supported')
        raise TypeError(msg)

@attrs.define(kw_only=True)
class TriaxialDatasetBuildRecipe:
    """Recipe for building triaxial datasets with required and optional attributes."""
    # Required attributes
    planestride: int | tuple[int, int, int]
    
    @classmethod
    def from_dict(cls, data_dict: dict):
        """
        Create a TriaxialDatasetBuildRecipe from a dictionary.
        Required attributes are enforced, while additional key-value pairs
        are set as attributes dynamically.
        """
        # Make a copy to avoid modifying the original
        data = data_dict.copy()
        
        # Create instance with required attributes
        instance = cls(planestride=data.pop('planestride'))
        
        # Set remaining attributes dynamically
        for key, value in data.items():
            setattr(instance, key, value)
        
        return instance


def create_recipe_from_spec(spec_dict: dict) -> dict:
    """
    Transform specification dictionaries to use attrs classes for structured sections.
    """
    result = spec_dict.copy()
    
    # Convert dataset_build_kwargs to TriaxialDatasetBuildRecipe
    if 'dataset_build_kwargs' in result:
        result['dataset_build_kwargs'] = TriaxialDatasetBuildRecipe.from_dict(
            result['dataset_build_kwargs']
        )
    
    return result


def combine(
    paths: list[Path],
    specification_mapping: Mapping[str, Mapping],
    defaults_mapping: Mapping[str, Mapping] | None = None,
) -> list[Mapping]:
    """
    Combine the ingredients to create a flat list of specifications that can be used
    to build datasets via the bulk_build_from_zarr method.
    """
    defaults_mapping = defaults_mapping or {}
    specmapping: defaultdict[str, list] = defaultdict(list)
    for path in paths:
        for internal_path, spec in specification_mapping.items():
            # Merge defaults with specific configuration
            merged_spec = {**defaults_mapping, **spec}
            # Process the dictionary to convert sections to attrs classes
            processed_spec = create_recipe_from_spec(merged_spec)
            
            final_spec = {
                'path': path,
                'internal_path': internal_path,
                **processed_spec
            }
            specmapping[internal_path].append(final_spec)
    return specmapping


def heal_groupwise(
    multisets: Mapping[Hashable, Sequence[torchdata.Dataset]],
    heal_kwargs: Mapping,
) -> Mapping[Hashable, Sequence[torchdata.Dataset]]:
    """
    Heal the multisets groupwise, i.e. every sequence that is expected
    to have similar shape.
    """
    healed_multisets = {}
    for key, datasets in multisets.items():
        healed_multisets[key] = healing.heal_datasets_3D(
            datasets,
            **heal_kwargs,
        )
    return healed_multisets


def to_loaders(multisets: Mapping[Hashable, Sequence[torchdata.Dataset]], batch_size: int, num_workers: int, shuffle: bool) -> Mapping[Hashable, torchdata.DataLoader]:
    """
    Convert the multisets to loaders.
    """
    loaders = {}
    for key, datasets in multisets.items():
        loaders[key] = torchdata.DataLoader(
            dataset=torchdata.ConcatDataset(datasets),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
    return loaders