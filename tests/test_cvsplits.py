import rich
import pytest

from woodnet.datasets.setup import INSTANCE_MAPPING, group_instances_by_class, convert_to_lists

from woodnet.cvsplits import INSTANCES, CLASSES, WOOD_CLASSES, ORIENTATION_ClASSES, remap_group

from woodnet.cvsplits import StratifiedKFoldsGenerator



def test():
    instance_by_class = group_instances_by_class(INSTANCE_MAPPING, format='mapping')

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

    rich.print(generator[0])