import pytest
import torch
import json

from functools import partial
from pathlib import Path

from woodnet.checkpoint.handlers import RWDHandler, generate_filename

MAGIC_WEIGHT_VALUE: float = 1701.0

def init(module, value):
    if type(module) == torch.nn.Linear:
        module.weight.fill_(value)

def check_magic_value(module):
    value = torch.tensor(MAGIC_WEIGHT_VALUE)
    if isinstance(module, torch.nn.Linear):
        if not torch.allclose(module.weight, value):
            raise ValueError('deviation from magic weight value')


@pytest.fixture(scope='module')
def model():
    net = torch.nn.Sequential(
        torch.nn.Linear(5, 25),
        torch.nn.ReLU(),
        torch.nn.Linear(25, 25),
        torch.nn.Linear(25, 2)
    )
    # intialize with magic value
    _init = partial(init, value=MAGIC_WEIGHT_VALUE)
    with torch.no_grad():
        net.apply(_init)
    return net


def test_generate_filename_prefix_no_qualifier():
    expected_prefix = 'romulan_star_empire'
    fname = generate_filename(prefix=expected_prefix)
    prefix, *rest = fname.split('-')
    assert prefix == expected_prefix


def test_generate_filename_prefix_with_qualifier():
    expected_prefix = 'romulan_star_empire'
    qualifier = 'superoptimal'
    fname = generate_filename(prefix=expected_prefix, qualifier=qualifier)
    prefix, *rest = fname.split('-')
    assert prefix == expected_prefix


def test_check_working_directory():
    basedir = Path('/path/to/checkpointdir')
    filepath = Path('/path/to/checkpointdir/picard.cptn')
    handler = RWDHandler(directory=basedir)
    handler.check_working_directory(path=filepath)


def test_check_working_directory_raises_exception():
    basedir = Path('/path/to/checkpointdir')
    filepath = Path('/path/to/filedir/picard.cptn')
    handler = RWDHandler(directory=basedir)

    with pytest.raises(ValueError):
        handler.check_working_directory(path=filepath)


def test_write_and_read(model, tmp_path):
    handler = RWDHandler(directory=tmp_path, prefix='test-chkpt')
    
    savepath = handler.write(model, qualifier='test-optimal')

    path_loaded_model = handler.read(path=savepath)
    name_loaded_model = handler.read(name=savepath.name)

    model.apply(check_magic_value)
    path_loaded_model.apply(check_magic_value)
    name_loaded_model.apply(check_magic_value)



def test_delete_properly_existing_file_by_name(model, tmp_path):
    handler = RWDHandler(directory=tmp_path, prefix='test_ckpt')
    savepath = handler.write(model=model, qualifier='test-optimal')
    handler.delete(name=savepath.name)


def test_delete_properly_existing_file_by_path(model, tmp_path):
    handler = RWDHandler(directory=tmp_path, prefix='test_ckpt')
    savepath = handler.write(model=model, qualifier='test-optimal')
    handler.delete(path=savepath)


def test_delete_nonexisting_file_by_path(tmp_path):
    handler = RWDHandler(directory=tmp_path)
    nonexisting_savepath = tmp_path / 'this-should-not-exist.pth'
    assert not nonexisting_savepath.exists(), 'file should not exist: test setup failure'
    with pytest.raises(FileNotFoundError):
        handler.delete(path=nonexisting_savepath)


def test_delete_nonexisting_file_by_name(tmp_path):
    handler = RWDHandler(directory=tmp_path)
    nonexisting_savepath = tmp_path / 'this-should-not-exist.pth'
    assert not nonexisting_savepath.exists(), 'file should not exist: test setup failure'
    with pytest.raises(FileNotFoundError):
        handler.delete(name=nonexisting_savepath.name)


def test_write_json(tmp_path):
    expected_data = {
        'who am i' : 'groot',
        'age' : 42,
        'ship' : {
            'name' : 'uss enterprise',
            'class' : 'sovereign',
            'phaser banks' : '5 mark 2',
            'torpedo launcher type' : 'quantum'
        }
    }
    filename = 'data-test.json'
    expected_filepath = tmp_path / filename
    handler = RWDHandler(directory=tmp_path)

    handler.write_json(expected_data, filename=filename)

    with expected_filepath.open(mode='r') as handle:
        data = json.load(fp=handle)

    assert data == expected_data, 'write <-> read data mismatch'