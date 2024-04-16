import torch
import pytest
import rich

from ruamel.yaml import YAML


from woodnet.datasets.volumetric_inference import (TransformedTileDatasetBuilder,
                                                   TransformedTileDataset)

from woodnet.inference.evaluate import evaluate, evaluate_multiple, Predictor
from woodnet.inference.inference import (create_parametrized_transforms,
                                         extract_IDs, extract_model_config)

from woodnet.models.volumetric import ResNet3D
from woodnet.inference.parametrized_transforms import (CongruentTransformList,
                                                       ParametrizedTransform,
                                                       generate_parametrized_transforms,
                                                       )


@pytest.fixture
def configuration() -> dict:
    """Provides an authentic training configuration."""
    data = """
    experiment_directory: /home/jannik/storage/trainruns-wood-2024/fullaug-256-bs-3-kf-sgd/fold-1
    device: cuda:2

    model:
        name: ResNet3D
        in_channels: 1
        compile:
            enabled: true
            dynamic: false
            fullgraph: false

    optimizer:
        name: SGD
        learning_rate: 1e-3
        momentum: 0.9
        nesterov: true

    loss:
        name: BCEWithLogitsLoss
        reduction: mean

    trainer:
        name: Trainer
        max_num_iters: 100000
        max_num_epochs: 1250
        validation_metric: ACC
        use_amp: true
        use_inference_mode: true
        log_after_iters: 1000
        save_model_checkpoint_every_n: 250
        validate_after_iters: 1500
        score_registry:
            name: Registry
            capacity: 5
            score_preference: higher_is_better
        parameter_logger:
            name: HistogramLogger

    loaders:
        dataset: TileDataset
        num_workers: 0
        batchsize: 3
        tileshape:
            - 256
            - 256
            - 256

        pin_memory: true
        train:
          instances_ID:
            - CT16
            - CT17
            - CT19
            - CT14
            - CT11
            - CT10
            - CT12
            - CT3
            - CT5
            - CT9
            - CT20
            - CT21
            - CT22
            
          transform_configurations:

            - name: Normalize3D
              mean: 110
              std: 950
            
            - name: EquiprobableSelector
              members:

                - name: Rotate90
                  dims:
                    - 0
                    - 1
                - name: Rotate90
                  dims:
                    - 1
                    - 2
                - name: Rotate90
                  dims:
                    - 0
                    - 2
            
            - name: EquiprobableSelector
              members:
                
                - name: GaussianNoise
                  p_execution: 0.33
                  mean: 0.0
                  std: 1.5

                - name: GaussianBlur
                  p_execution: 0.33
                  ksize: 5
                  sigma: 1

                - name: GaussianBlur
                  p_execution: 0.33
                  ksize: 5
                  sigma: 3

        val:
          instances_ID:
            - CT18
            - CT2
            - CT13 
            - CT7 
            - CT6 
            - CT8
            - CT15

          transform_configurations:

            - name: Normalize3D
              mean: 110
              std: 950
    """
    yaml = YAML()
    conf = yaml.load(data)
    return conf





@pytest.fixture
def smooth_transforms() -> list[ParametrizedTransform]:
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
    return transforms


@pytest.fixture
def noise_transforms() -> list[ParametrizedTransform]:
    specification = {
        'name' : 'GibbsNoiser',
        'class_name' : 'GibbsNoise',
        'parameters' : [
            {'alpha' : 0.3},
            {'alpha' : 0.6},
            {'alpha' : 0.9}
        ]
    }
    parametrizations = generate_parametrized_transforms(specification)
    return parametrizations


@pytest.fixture(scope='function')
def datasets() -> list[TransformedTileDataset]:
    N: int = 2
    ID: list[str] = ['CT10', 'CT9'] 
    builder = TransformedTileDatasetBuilder()
    datasets = builder.build(instances_ID=ID, tileshape=(64, 64, 64), transform_configurations=None)
    assert len(datasets) == N
    return datasets


@pytest.fixture
def test_loader(datasets):
    # construct cheap data loader
    batch_size: int = 64
    dataset = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader



class Test_evaluate:
    @pytest.mark.slow
    def test_smoke_basic(self, datasets, smooth_transforms):
        import rich
        device = torch.device('cuda:0')
        dtype = torch.float32
        model = ResNet3D(in_channels=1)
        # move model
        model.to(device=device, dtype=dtype)
        # just select one dataset for performance reasons
        dataset = datasets[0]
        assert isinstance(dataset, TransformedTileDataset), 'required for test setup'
        # construct cheap data loader
        batch_size: int = 64
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        result = evaluate(model,
                          parametrizations=smooth_transforms,
                          loader=loader,
                          device=device,
                          dtype=dtype,
                          display_loader_progress=True,
                          display_parametrizations_progress=True
                          )
        rich.print(result)

        

class Test_Predictor:
    @pytest.mark.slow
    def test_smoke_predict_method(self, test_loader, smooth_transforms, noise_transforms):
        """Basic smoke test: method should run through and provide a results dictionary."""
        device = torch.device('cuda:0')
        dtype = torch.float32
        use_amp = True
        use_inference_mode = True
        loader = test_loader
        model = ResNet3D(in_channels=1)
        gauss_parametrizations = CongruentTransformList(smooth_transforms)
        gibbs_parametrizations = CongruentTransformList(noise_transforms)
        transforms = [gauss_parametrizations, gibbs_parametrizations]

        predictor = Predictor(model,
                              device=device,
                              dtype=dtype,
                              use_amp=use_amp,
                              use_inference_mode=use_inference_mode,
                              display_transforms_progress=True,
                              display_parametrizations_progress=True,
                              display_loader_progress=True
                              )

        report = predictor.predict(loader=loader, transforms=transforms)

        # results
        import rich
        rich.print(report)




class Test_evaluate_multiple:

    @pytest.mark.slow
    def test_smoke(self, smooth_transforms, noise_transforms, test_loader):
        """Basal test that the function runs for a correct setup."""
        device = torch.device('cuda:0')
        dtype = torch.float32
        use_amp = True
        use_inference_mode = True
        loader = test_loader
        models_mapping = {f'mock_checkpoint-{i}' : ResNet3D(in_channels=1) for i in range(2)}
        gauss_parametrizations = CongruentTransformList(smooth_transforms)
        gibbs_parametrizations = CongruentTransformList(noise_transforms)
        transforms = [gauss_parametrizations, gibbs_parametrizations]

        results = evaluate_multiple(models_mapping=models_mapping,
                                    loader=loader,
                                    transforms=transforms,
                                    device=device, dtype=dtype, use_amp=use_amp,
                                    use_inference_mode=use_inference_mode,
                                    display_model_instance_progress=True,
                                    leave_model_instance_progress=True,
                                    display_transforms_progress=True,
                                    display_parametrizations_progress=True,
                                    display_loader_progress=True)

        rich.print(results)




class Test_create_parametrized_transforms:
    def test_smoke(self):
        specifications = [
            {
                'name' : 'GaussianSmoothie',
                'class_name' : 'GaussianSmooth',
                'parameters' : [
                    {'sigma' : 1.0},
                    {'sigma' : 2.0},
                ]
            },
            {
                'name' : 'MedianMan',
                'class_name' : 'MedianSmooth',
                'parameters' : [
                    {'radius' : 1},
                    {'radius' : 2}
                ]
            }
        ]
        config = {'junk' : 'food', 'captain' : 'picard',
                  'parametrized_transforms' : specifications}
        parametrizations = create_parametrized_transforms(config)

        import rich
        rich.print(parametrizations)
    
    def test_returns_none_on_fully_missing_parametrized_transforms_spec(self):
        config = {'foo' : 'bar', 'baz' : 1}
        parametrizations = create_parametrized_transforms(config)
        assert parametrizations is None

    @pytest.mark.parametrize('spec', ([], None))
    def test_returns_none_on_disabled_parametrized_transforms_spec(self, spec):
        config = {'foo' : 'bar', 'baz' : 1, 'parametrized_transform' : spec}
        parametrizations = create_parametrized_transforms(config)
        assert parametrizations is None


class Test_extract_IDs:
    def test_basic_smoke(self, configuration):
        expected_IDs = ['CT18', 'CT2', 'CT13', 'CT7', 'CT6', 'CT8', 'CT15']
        retrieved_IDs = extract_IDs(configuration)
        assert expected_IDs == retrieved_IDs
        


class Test_extract_model_config:

    def test_basic_extraction(self, configuration):
        modelconf, compileconf = extract_model_config(configuration)
        assert modelconf == {'name': 'ResNet3D', 'in_channels': 1}
        assert compileconf == {'enabled' : True, 'dynamic' : False, 'fullgraph' : False}
