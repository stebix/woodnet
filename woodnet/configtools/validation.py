"""
Provides pydantic model patterns for systematic validation approach fo
configuration dictionary data to process them via downstream CLI.

@jsteb 2023
"""
import pydantic

from typing import Union, Literal

from typing_extensions import Annotated
from annotated_types import Gt

from woodnet.custom.types import PathLike


class Config(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra='allow')


class Model(Config):
    name: str


class Optimizer(Config):
    name: str


class Loss(Config):
    name: str


class Trainer(Config):
    name: str
    max_num_epochs: int
    max_num_iters: int
    validation_metric: str | None = None
    use_amp: bool | None = True
    use_inference_mode: bool | None = True
    log_after_iters: Annotated[int, Gt(0)] | None = 1000
    save_model_checkpoint_every_n: Annotated[int, Gt(0)] | None = 10000


class Predictor(Config):
    name: str
    use_amp: bool | None = True
    use_inference_mode: bool | None = True
    


class Transformation(Config):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class TrainingLoader(Config):
    instances_ID: list[str]
    transform_configurations: list[Transformation] | None = None


class ValidationLoader(Config):
    instances_ID: list[str]
    transform_configurations: list[Transformation] | None = None


class PlanarLoaders(Config):
    """Training and validation loaders for 2D planar setup."""
    dataset: Literal['EagerSliceDataset', 'SliceDataset']
    axis: int
    num_workers: int | None = None
    batchsize: int | None = None
    pin_memory: bool | None = None

    train: TrainingLoader
    val: ValidationLoader

    def dataset_kwargs(self) -> dict:
        return {'axis' : self.axis}


class TriaxialLoaders(Config):
    """Training and validation loaders for 2.5D triaxial setup."""
    dataset: Literal['TriaxialDataset']
    planestride: tuple[int, int, int]
    num_workers: int | None = None
    batchsize: int | None = None
    tileshape: tuple[int, int, int] | None = None
    pin_memory: bool | None = None

    train: TrainingLoader
    val: ValidationLoader

    def dataset_kwargs(self) -> dict:
        return {'planestride' : self.planestride, 'tileshape' : self.tileshape}


class VolumetricLoaders(Config):
    """Training and validation loaders for 3D volumetric setup."""
    dataset: Literal['TileDataset']
    num_workers: int | None = None
    batchsize: int | None = None
    tileshape: tuple[int, int, int] | None = None
    pin_memory: bool | None = None

    train: TrainingLoader
    val: ValidationLoader

    def dataset_kwargs(self) -> dict:
        return {'tileshape' : self.tileshape}



class TrainingConfiguration(Config):
    """Fully validated training configuration."""
    experiment_directory: str
    device: str
    model: Model
    optimizer: Optimizer
    loss: Loss
    trainer: Trainer
    loaders: Union[PlanarLoaders,
                   TriaxialLoaders,
                   VolumetricLoaders] = pydantic.Field(..., discriminator='dataset')



class PredictionConfiguration(Config):
    trained_model_path: str
    target_directory: str
    device: str
    result_dict_filename: str

    model: Model
    predictor: Predictor

