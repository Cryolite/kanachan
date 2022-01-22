#!/usr/bin/env python3

from pathlib import Path
import logging
from typing import (Optional, Union,)
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
