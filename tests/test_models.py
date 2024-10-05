import sys
import torch

import pytest

from woodnet.models import (create_model, collect_custom_model_modules,
                            get_model_class, CANONICAL_MODEL_MODULES)
from woodnet.models.planar import ResNet18
from woodnet.models.volumetric import ResNet3D




def test_collect_custom_model_modules(tmp_path):
    # create dummy custom module files and dummy __init__.py file
    dummy_files = [
        tmp_path / 'dummy.py',
        tmp_path / 'customcontrib_dummy1.py',
        tmp_path / 'customcontrib_dummy2.py',
        tmp_path / 'falseprefix_dummy3.py'
    ]
    for file in dummy_files:
        file.write_text('dummy content')

    custom_modules = list(collect_custom_model_modules(tmp_path))
    assert set(custom_modules) == {'customcontrib_dummy1', 'customcontrib_dummy2'}
    


def test_get_model_from_custom_module(tmp_path):
    # create dummy custom module files and dummy __init__.py file
    dummy1_content = 'class ModelDummy1_Picard:\n    pass'
    dummy2_content = 'class ModelDummy2_Sisko:\n    pass'
    
    dmod1 = tmp_path / 'customcontrib_dummy1.py'
    dmod2 = tmp_path / 'customcontrib_dummy2.py'

    with dmod1.open('w') as f:
        f.write(dummy1_content)
    with dmod2.open('w') as f:
        f.write(dummy2_content)

    # hack to add the temporary path to the module search path
    sys.path.append(str(tmp_path))

    modules = [
        *CANONICAL_MODEL_MODULES,
        *[elem.stem for elem in (dmod1, dmod2)]
    ]
    conf1 = {'name' : 'ModelDummy1_Picard'}
    model_class1 = get_model_class(conf1, modules=modules)
    assert model_class1.__name__ == 'ModelDummy1_Picard'



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