import rich
from pathlib import Path

import pytest

from woodnet.configtools import load_yaml

from woodnet.datasets.setup import DataConfiguration, group_instances_by_class, load_data_configuration

FPATH = Path('/home/jannik/code/woodnet/tests/_dataconf.yaml')

@pytest.fixture
def dataconf_raw():
    return load_yaml(FPATH)


def test_load_data_configuration() -> None:
    dataconf = load_data_configuration(FPATH)
    rich.print(dataconf)



def test_correct_grouping(dataconf_raw):
    dataconf = DataConfiguration(**dataconf_raw)

    result = group_instances_by_class(dataconf.instance_mapping)

    pinusID = [3, 5, 7, 6, 8, 9, 15, 20, 21, 22]
    expected_pinus_set = {f'CT{ID}' for ID in pinusID}
    acerID = [16, 17, 19, 18, 14, 11, 2, 10, 12, 13]
    expected_acer_set = {f'CT{ID}' for ID in acerID}

    assert set(result['pinus']) == expected_pinus_set
    assert set(result['acer']) == expected_acer_set