import rich
import pytest

from woodnet.datasets.setup import group_instances_by_class, convert_to_lists, InstanceFingerprint

from woodnet.cvsplits import StratifiedKFoldsGenerator

from tests.scaffolding.syntheticdata import (ClassSpecification,
                                             create_data_configuration)

@pytest.mark.skip
def test(monkeypatch):

    specs = [
        ClassSpecification(name='acer', label=0, groups={'red', 'green'}, instances_per_group=3),
        ClassSpecification(name='pinus', label=1, groups={'blue', 'green'}, instances_per_group=3),
    ]
    data_configuration = create_data_configuration(specs, base_location='path/to/data', internal_path='group/dset')
    instance_mapping = {k: InstanceFingerprint(**v) for k, v in data_configuration['instance_mapping'].items()}

    _ = group_instances_by_class(instance_mapping, format='mapping')

    instances, classes, groups = convert_to_lists(instance_mapping)

    monkeypatch.setattr('woodnet.cvsplits.INSTANCES', instances)
    monkeypatch.setattr('woodnet.cvsplits.WOOD_CLASSES', classes)
    monkeypatch.setattr('woodnet.cvsplits.ORIENTATION_CLASSES', groups)


    generator = StratifiedKFoldsGenerator()
    r = generator[1]
    
    rich.print(r)

