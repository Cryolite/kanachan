#!/usr/bin/env python3

import re
from pathlib import Path
import logging
from typing import NoReturn, Optional, Union
import torch
from torch import nn
from torch.utils.data import IterableDataset


def initialize_logging(
        experiment_path: Path, local_rank: Optional[int]) -> None:
    fmt = '%(asctime)s %(filename)s:%(lineno)d:%(levelname)s: %(message)s'
    if local_rank is None:
        path = experiment_path / 'training.log'
    else:
        path = experiment_path / f'training.{local_rank}.log'
    file_handler = logging.FileHandler(path, encoding='UTF-8')
    if local_rank is None or local_rank == 0:
        console_handler = logging.StreamHandler()
        handlers = (console_handler, file_handler)
    else:
        handlers = (file_handler,)
    logging.basicConfig(format=fmt, level=logging.INFO, handlers=handlers)


def load_state_dict(module: nn.Module, state_dict: dict) -> None:
    fixed_state_dict = {}
    for k, v in state_dict.items():
        k = re.sub('^.*?__', '', k)
        k = re.sub('\\..*?__', '.', k)
        k = re.sub('^module\\.', '', k)
        fixed_state_dict[k] = v
    module.load_state_dict(fixed_state_dict)


class Dataset(IterableDataset):
    def __init__(self, path: Union[str, Path], iterator_adaptor) -> None:
        super(Dataset, self).__init__()
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise RuntimeError(f'{path}: does not exist.')
        self.__path = path
        self.__iterator_adaptor = iterator_adaptor

    def __iter__(self):
        return self.__iterator_adaptor(self.__path)

    def __getitem__(self, index) -> NoReturn:
        raise NotImplementedError('Not implemented.')


def get_gradient(model: nn.Module) -> torch.Tensor:
    gradient = [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
    return torch.cat(gradient)


def is_gradient_nan(model: nn.Module) -> bool:
    gradient = get_gradient(model)
    return torch.any(torch.isnan(gradient)).item()
