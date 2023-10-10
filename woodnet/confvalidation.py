"""
Provides pydantic model patterns for systematic validation approach fo
configuration dictionary data to process them via downstream CLI.

@jsteb 2023
"""
import pathlib
import pydantic

from typing_extensions import Annotated
from annotated_types import Gt

from woodnet.custom.types import PathLike


class Config(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


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
    use_amp: bool | None = None
    use_inference_mode: bool | None = None
    log_after_iters: Annotated[int, Gt(0)] | None = None
    save_model_checkpoints_every_n: Annotated[int, Gt(0)] | None = None


class Transformation(Config):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class TrainingLoader(Config):
    instances_ID: list[str]
    transform_configurations: list[Transformation] | None = None


class ValidationLoader(Config):
    instances_ID: list[str]
    transform_configurations: list[Transformation] | None = None


class Loaders(Config):
    """Training and validation loaders."""
    dataset: str
    num_workers: int | None = None
    batchsize: int | None = None
    tileshape: tuple[int, int, int] | None = None
    pin_memory: bool | None = None

    train: TrainingLoader
    val: ValidationLoader


class TrainingConfiguration(Config):
    """Fully validated training configuration."""
    experiment_directory: PathLike
    device: str
    model: Model
    optimizer: Optimizer
    loss: Loss
    trainer: Trainer
    loaders: Loaders

    @pydantic.validator('experiment_directory')
    def to_path(cls, v: str) -> pathlib.Path:
        return pathlib.Path(v)



