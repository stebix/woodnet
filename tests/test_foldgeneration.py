from pathlib import Path

from woodnet.configtools import load_yaml
from woodnet.configtools.foldgeneration import *


def test_initial():
    folds = generate_stratified_kfolds()

    print(folds)



def test_refold():
    thisfile = Path(__file__)
    confpath = thisfile.parents[1] / 'woodnet/trainconf.yaml'
    conf = load_yaml(confpath)

    result = refold_configuration(conf, strategy='stratified_group_kfold', foldnum=3)

    print('prerefold')
    print(conf['loaders']['val'])

    print('\n\npostrefold')
    print(result.loaders.val)



    