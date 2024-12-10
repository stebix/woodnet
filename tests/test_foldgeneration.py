from pathlib import Path

import pytest

from woodnet.configtools import load_yaml
from woodnet.configtools.foldgeneration import refold_configuration
from woodnet.cvsplits import strategy_to_generator, CVStrategy

# TODO: Reform tests for the fold generation

@pytest.mark.parametrize('strategy', ['stratified_group_kfold', 'stratified_kfold'])
def test_initial(strategy):
    strategy = CVStrategy(strategy)
    generator = strategy_to_generator[strategy]()
    folds = generator[1]
    print(folds)


@pytest.mark.skip
def test_refold():
    thisfile = Path(__file__)
    confpath = thisfile.parents[1] / 'woodnet/trainconf.yaml'
    conf = load_yaml(confpath)

    result = refold_configuration(conf, strategy='stratified_group_kfold', foldnum=3)

    print('prerefold')
    print(conf['loaders']['val'])

    print('\n\npostrefold')
    print(result.loaders.val)



    