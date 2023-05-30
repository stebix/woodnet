import datetime
import torch

from custom.types import PathLike


class IOHandler:

    def __init__(self, directory: PathLike) -> None:
        self.directory = directory

    def setup(self):
        pass




class Evaluator:

    def __init__(self,
                 transformer_config,
                 io_handler: IOHandler) -> None:
        self.transformer_config: list[dict] = transformer_config
        self.io_handler = io_handler


    @staticmethod
    def get_timestamp(fmt: str | None = None) -> str:
        fmt = fmt or '%Y-%m%d_%H-%M-%S'
        return datetime.datetime.now.strftime(fmt)


