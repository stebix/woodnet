import logging

from ruamel.yaml import YAML

from woodnet.models import create_model
from woodnet.train import create_optimizer, create_loss, create_loaders, run_training_experiment, retrieve_logged


def test_retrieve_logged(caplog):
    caplog.set_level(logging.DEBUG)
    data = {
        'setting' : True,
        'parameter' : 'warp 9'
    }
    value_a = retrieve_logged(data, key='energize', default='do not beam me up', method='get')
    value_b = retrieve_logged(data, key='parameter', default='warp 1', method='get')

    assert value_a == 'do not beam me up'
    assert value_b == 'warp 9'



def test_run_training_experiment_smoke():
    p = '/home/jannik/code/woodnet/woodnet/trainconf.yaml'
    run_training_experiment(p)


def test_train_smoke():
    yaml = YAML(typ='safe')
    with open('/home/jannik/code/woodnet/woodnet/trainconf.yaml') as handle:
        conf = yaml.load(handle)


    model = create_model(conf)

    opt = create_optimizer(model, conf)

    print(opt)



def test_create_loss():
    yaml = YAML(typ='safe')
    with open('/home/jannik/code/woodnet/woodnet/trainconf.yaml') as handle:
        conf = yaml.load(handle)

    loss = create_loss(conf)

    print(loss)



def test_create_loaders():
    yaml = YAML(typ='safe')
    with open('/home/jannik/code/woodnet/woodnet/trainconf.yaml') as handle:
        conf = yaml.load(handle)

    loaders = create_loaders(conf)

    print(loaders)
