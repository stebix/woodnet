import numpy as np
import torch
import attrs

import pytest

import woodnet.datasets.triaxial.v3 as triaxial

import rich.console


def test_smote_generate_triaxial_planes_parameter():
    shape = (512, 512, 512)
    planestride = (75, 75, 75)

    axiswise = []
    for axis, stride in enumerate(planestride):
        axiswise.append(
            triaxial.generate_plane_indices(shape, axis, stride)
        )
    
    op = triaxial.row_outer_product(axiswise)

    slices = []
    for elems in op:
        slices.append(
            np.apply_along_axis(func1d=triaxial.convert_to_slices_from, axis=1, arr=elems)
        )


@attrs.define
class MockFingerprint:
    """
    Mock class for fingerprint.
    """
    class_: str = 'acer'
    id: str = 'test_id'


class Test_V3TriaxialDataset:
    
    @pytest.mark.parametrize(
        'shape',
        ((32, 32, 32), (66, 66, 66), (77, 77, 77))
    )
    def test_smoke_initializtion(self, shape):
        # Create a small 3D volume for testing
        rng = np.random.default_rng(42)
        data = rng.normal(size=shape).astype(np.float32)
        
        # Define basic parameters for the dataset
        planestride = (16, 16, 16)
        fingerprint = MockFingerprint()
        stats = {'mean': 0.5, 'std': 0.1}
        
        # Instantiate the dataset
        dataset = triaxial.V3TriaxialDataset(
            phase='train',
            data=data,
            planestride=planestride,
            fingerprint=fingerprint,
            stats=stats,
            classlabel_mapping={'acer': 0},
        )
        
        # Check dataset was created successfully
        assert dataset is not None
        
        # Get an element from the dataset
        element, lbl = dataset[0]  # Assuming the second return value isn't needed for this test
        
        # Check the element type and shape
        assert isinstance(element, torch.Tensor), 'Dataset should return a torch.Tensor'
        assert isinstance(lbl, torch.Tensor), 'Dataset should return a torch.Tensor label'
        assert element.ndim == 3, 'Returned tensor should be 3D (C, H, W)'
        assert element.shape[0] == 3, 'Channel dimension should be 3 for triaxial planes'
        assert element.shape == (3, shape[-2], shape[-1]), 'Returned tensor shape should be (3, H, W)'
        
        # Print additional info for debugging
        print(f'Element shape: {element.shape}')
