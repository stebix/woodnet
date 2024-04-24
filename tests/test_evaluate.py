from collections.abc import Sequence

import pytest
import rich

from woodnet.evaluate import (load_parametrized_transforms_specs_from,
                              generate_transforms_from,
                              PRESETS_DIRECTORY)


class Test_load_parametrized_transforms_specs_from_preset:

    def test_smoke_load_basal_preset(self):
        preset_name = 'basal-smoothing'
        preset = load_parametrized_transforms_specs_from(preset_name)
        rich.print(preset)


class Test_generate_transforms_from_preset_name:

    def test_smoke_generate_from_basal_preset(self):
        preset_name = 'basal-smoothing'
        transforms = generate_transforms_from(preset_name)
        rich.print(transforms)


def test_integrity_validity_of_all_preset_files():
    """
    Validity and integrity test of *ALL* JSON preset files present in
    the designated directory.
    Successfull integrity tests means that the generating function
    produces transforms error-free
    into a `Sequence[CongruentParametrizations]`
    """
    presets = []
    for item in PRESETS_DIRECTORY.iterdir():
        if item.suffix.endswith('.json'):
            presets.append(item.name)
        
    for preset_name in presets:
        transforms = generate_transforms_from(preset_name)
        assert isinstance(transforms, Sequence)
