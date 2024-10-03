import torch

import pytest

from woodnet.models import create_model
from woodnet.models.planar import ResNet18
from woodnet.models.volumetric import ResNet3D


def test_2D_ResNet18_via_create_model():
    """Basic smoke test if model creation and forward pass go through."""
    in_channels = 1
    out_channels = 1
    batch_size = 3
    size = (batch_size, in_channels, 256, 256)
    inputs = torch.randn(size=size)

    model_conf = {
        'name' : 'ResNet18',
        'in_channels' : in_channels
    }
    # create_model expects a top-level configuration dictionary with a 'model' key
    conf = {'model' : model_conf}
    model = create_model(conf)

    with torch.no_grad():
        prediction = model(inputs)
    
    assert prediction.shape[0] == batch_size
    assert prediction.shape[1] == out_channels



def test_2D_ResNet18_via_direct_create():
    """Test direct instantiation with modification of class variables."""
    in_channels = 1
    out_channels = 3
    batch_size = 3
    size = (batch_size, in_channels, 256, 256)
    inputs = torch.randn(size=size)


    ResNet18.num_classes = out_channels

    kwargs = {
        'in_channels' : in_channels,
        'final_nonlinearity' : 'softmax'
    }
    model = ResNet18(**kwargs)

    with torch.no_grad():
        prediction = model(inputs)
    
    assert prediction.shape[0] == batch_size
    assert prediction.shape[1] == out_channels
    assert isinstance(model.final_nonlinearty, torch.nn.Softmax)



def softcheck_is_compiled(obj, eager_base_class: type | None = None) -> bool:
    """
    Brittle test whether the given model `obj` is compiled/optimized.
    TODO: depends on private code structure of PyTorch and may break easily.
    """
    if eager_base_class:
        eager_base_class_name = eager_base_class.__name__
        try:
            orig_mod_name = obj._orig_mod.__class__.__name__
        except AttributeError:
            # If no _orig_mod attribute is present, we currently accept this as
            # an indicator that this object is not an torch OptimizedModel 
            return False
        names_match = eager_base_class_name == orig_mod_name
    else:
        names_match = True 
    return 'OptimizedModule' in str(type(obj)) and names_match



class Test_create_model_compilation_opts:
    """
    Test the more involved control flow of the create_model function
    concerning the compilation options.
    """
    @pytest.mark.parametrize(
            'name,classobj',
            [('ResNet3D', ResNet3D), ('ResNet18', ResNet18)])
    def test_with_no_compilation_configuration(self, name, classobj):
        conf = {
            'model' : {
                'name' : name,
                'in_channels' : 1
            }
        }
        model = create_model(conf)
        assert isinstance(model, classobj)
        assert not softcheck_is_compiled(model)

    @pytest.mark.parametrize(
            'name,classobj',
            [('ResNet3D', ResNet3D), ('ResNet18', ResNet18)])
    def test_with_enabled_compilation_configuration(self, name, classobj):
        conf = {
            'model' : {
                'name' : name,
                'in_channels' : 1,
                'compile' : {
                    'enabled' : True,
                    'fullgraph' : False,
                    'dynamic' : False
                }
            }
        }
        model = create_model(conf)
        assert softcheck_is_compiled(model, eager_base_class=classobj)


    @pytest.mark.parametrize(
            'name,classobj',
            [('ResNet3D', ResNet3D), ('ResNet18', ResNet18)])
    def test_with_enabled_compilation_configuration_but_no_compile_override(self, name, classobj):
        conf = {
            'model' : {
                'name' : name,
                'in_channels' : 1,
                'compile' : {
                    'enabled' : True,
                    'fullgraph' : False,
                    'dynamic' : False
                }
            }
        }
        model = create_model(conf, no_compile_override=True)
        assert isinstance(model, classobj)
        assert not softcheck_is_compiled(model, eager_base_class=classobj)


    @pytest.mark.parametrize(
            'name,classobj',
            [('ResNet3D', ResNet3D), ('ResNet18', ResNet18)])
    def test_with_disabled_compilation_configuration_and_no_compile_override(self, name, classobj):
        conf = {
            'model' : {
                'name' : name,
                'in_channels' : 1,
                'compile' : {
                    'enabled' : False,
                    'fullgraph' : False,
                    'dynamic' : False
                }
            }
        }
        model = create_model(conf, no_compile_override=True)
        assert isinstance(model, classobj)
        assert not softcheck_is_compiled(model, eager_base_class=classobj)