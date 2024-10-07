import torch
import pytest

from woodnet.inference.parametrized_transforms import (get_transform_class,
                                                       ParametrizedTransform,
                                                       generate_parametrized_transforms,
                                                       CongruentTransformList)



class Test_ParametrizedTransform:

    def test_create_and_has_raw_data_passthrough(self):
        x = torch.randn((1, 8, 8, 8))
        identity_ptf = ParametrizedTransform.make_identity()
        out = identity_ptf.transform(x)
        assert torch.allclose(x, out)

    def test_name_is_set(self):
        external_name = 'no_change'
        identity_ptf = ParametrizedTransform.make_identity(name=external_name)
        assert identity_ptf.name == external_name
        



class Test_get_transform_class:

    def test_with_monai_transform(self):
        from monai.transforms.intensity.array import GaussianSmooth
        name = 'GaussianSmooth'
        class_ = get_transform_class(name)
        assert class_ == GaussianSmooth
        
    
    def test_with_internal_transform(self):
        from woodnet.transformations.transforms import Normalize3D
        name = 'Normalize3D'
        class_ = get_transform_class(name)
        assert class_ ==  Normalize3D

    
    def test_failure_for_nonexisting_transform(self):
        name = 'UnicornTransform'
        with pytest.raises(AttributeError):
            _ = get_transform_class(name)



class Test_generate_parametrized_transforms:

    def test_with_correct_specification(self):
        from monai.transforms.intensity.array import GaussianSmooth
        specification = {
            'name' : 'GaussianSmoothie',
            'class_name' : 'GaussianSmooth',
            'parameters' : [
                {'sigma' : 1.0},
                {'sigma' : 2.0},
                {'sigma' : 3.0}
            ]
        }
        transforms = generate_parametrized_transforms(specification)
        assert len(transforms) == 3
        for transform in transforms:
            assert isinstance(transform, ParametrizedTransform)
            assert transform.name == specification['name']
            assert transform.parameters.keys() == set(['sigma'])
            assert isinstance(transform.transform, GaussianSmooth)

    
    def test_with_multiple_specifications(self):
        from monai.transforms.intensity.array import GaussianSmooth, MedianSmooth
        spec_gauss = {
            'name' : 'GaussianSmoothie',
            'class_name' : 'GaussianSmooth',
            'parameters' : [
                {'sigma' : 1.0},
                {'sigma' : 2.0},
                {'sigma' : 3.0}
            ]
        }
        spec_median = {
            'name' : 'MedianMeister',
            'class_name' : 'MedianSmooth',
            'parameters' : [
                {'radius' : 1},
                {'radius' : 2},
                {'radius' : 3}
            ]
        }
        specifications = [spec_gauss, spec_median]
        transforms = generate_parametrized_transforms(*specifications)
        assert len(transforms) == 2
        gaussian_ptfs = transforms[0]
        median_ptfs = transforms[1]

        for tf in gaussian_ptfs:
            assert tf.name == 'GaussianSmoothie'
            assert isinstance(tf.transform, GaussianSmooth)

        for tf in median_ptfs:
            assert tf.name == 'MedianMeister'
            assert isinstance(tf.transform, MedianSmooth)



class Test_CoherentList:


    def test_initialization_from_vanilla_list(self):
        # generate from specification and insert into CoherentList
        # should work
        specification = {
            'name' : 'GaussianSmoothie',
            'class_name' : 'GaussianSmooth',
            'parameters' : [
                {'sigma' : 1.0},
                {'sigma' : 2.0},
                {'sigma' : 3.0}
            ]
        }
        transforms = generate_parametrized_transforms(specification)
        coherent_list = CongruentTransformList(transforms)
        assert len(coherent_list) == 3


    def test_failure_on_appending_nonconforming_item(self):
        from monai.transforms.intensity.array import MedianSmooth
        specification = {
            'name' : 'GaussianSmoothie',
            'class_name' : 'GaussianSmooth',
            'parameters' : [
                {'sigma' : 1.0},
                {'sigma' : 2.0},
                {'sigma' : 3.0}
            ]
        }
        transforms = generate_parametrized_transforms(specification)
        coherent_list = CongruentTransformList(transforms)
        # create another valid but nonconoforming transformation
        median_smooth = MedianSmooth(radius=1)
        ptf = ParametrizedTransform(name='MedianSmooth', parameters={'radius' : 1}, transform=median_smooth)
        with pytest.raises(ValueError):
            coherent_list.append(ptf)


    def test_reinitialization_upon_populating_and_depopulating(self):
        """
        When the list is emptied, full newly-shaped ParametrizedTransform instances
        should be accepted. I.e. emptying resets the container to a basal state.
        """
        from monai.transforms.intensity.array import MedianSmooth
        specification = {
            'name' : 'GaussianSmoothie',
            'class_name' : 'GaussianSmooth',
            'parameters' : [
                {'sigma' : 1.0},
                {'sigma' : 2.0},
                {'sigma' : 3.0}
            ]
        }
        transforms = generate_parametrized_transforms(specification)
        coherent_list = CongruentTransformList(transforms)
        for _ in range(3):
            coherent_list.pop()
        # create another valid but nonconoforming transformation
        median_smooth = MedianSmooth(radius=1)
        ptf = ParametrizedTransform(name='MedianSmooth', parameters={'radius' : 1}, transform=median_smooth)
        coherent_list.append(ptf)
        assert len(coherent_list) == 1



