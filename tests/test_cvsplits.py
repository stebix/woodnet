import rich
import pytest

from woodnet.datasets.setup import group_instances_by_class, convert_to_lists, InstanceFingerprint

from woodnet.cvsplits import INSTANCES, WOOD_CLASSES, remap_group, StratifiedKFoldsGenerator

from tests.scaffolding.syntheticdata import (ClassSpecification,
                                             create_data_configuration, create_instance_informations)

@pytest.mark.skip
def test(monkeypatch):

    specs = [
        ClassSpecification(name='acer', label=0, groups={'red', 'green'}, instances_per_group=3),
        ClassSpecification(name='pinus', label=1, groups={'blue', 'green'}, instances_per_group=3),
    ]
    data_configuration = create_data_configuration(specs, base_location='path/to/data', internal_path='group/dset')
    instance_mapping = {k: InstanceFingerprint(**v) for k, v in data_configuration['instance_mapping'].items()}

    instance_by_class = group_instances_by_class(instance_mapping, format='mapping')

    instances, classes, groups = convert_to_lists(instance_mapping)

    monkeypatch.setattr('woodnet.cvsplits.INSTANCES', instances)
    monkeypatch.setattr('woodnet.cvsplits.WOOD_CLASSES', classes)
    monkeypatch.setattr('woodnet.cvsplits.ORIENTATION_CLASSES', groups)


    generator = StratifiedKFoldsGenerator()
    r = generator[1]
    
    rich.print(r)

    raise KeyError

    nid, ncl, ngr = convert_to_lists(INSTANCE_MAPPING)

    rich.print(INSTANCES)
    rich.print(WOOD_CLASSES)

    old = {i : c for i, c in zip(INSTANCES, WOOD_CLASSES)}
    new = {i : c for i, c in zip(nid, ncl)}

    assert old == new

    rich.print(nid)
    rich.print(ncl)

    mapping = {'axial' : 'axiallike', 'axial-tangential' : 'axiallike'}
    remapped = remap_group(ngr,
                           mapping=mapping)

    rich.print(
        [(pre, post) for pre, post in zip(ngr, remapped)]
    )


    generator = StratifiedKFoldsGenerator()

    rich.print(generator[1])