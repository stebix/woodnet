from collections import defaultdict
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from woodnet.datasets.buildtools import healing

def combine(
    paths: list[Path],
    specification_mapping: Mapping[str, Mapping],
    defaults_mapping: Mapping[str, Mapping] | None = None,
) -> list[Mapping]:
    """
    Combine the ingredients to creata a flat list of specifications that can be used
    to build datasets via the bulk_build_from_zarr method.
    """
    defaults_mapping = defaults_mapping or {}
    specmapping: defaultdict[str, Mapping] = defaultdict(list)
    for path in paths:
        for internal_path, spec in specification_mapping.items():
            kwargs = defaults_mapping | spec
            spec = {
                'path' : path,
                'internal_path' : internal_path,
                **kwargs
            }
            specmapping[internal_path].append(spec)
    return specmapping

def heal_groupwise(
    multisets: Mapping[Hashable, Sequence[Dataset]],
    heal_kwargs: Mapping,
) -> Mapping[Hashable, Sequence[Dataset]]:
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


def to_loaders(multisets: Mapping[Hashable, Sequence[Dataset]], batch_size: int, num_workers: int, shuffle: bool) -> Mapping[Hashable, torch.utils.data.DataLoader]:
    """
    Convert the multisets to loaders.
    """
    loaders = {}
    for key, datasets in multisets.items():
        loaders[key] = torch.utils.data.DataLoader(
            dataset=torch.utils.data.ConcatDataset(datasets),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
    return loaders