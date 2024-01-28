from pathlib import Path
from typing import NoReturn, Iterable
from torch.utils.data import IterableDataset


class Dataset(IterableDataset):
    def __init__(self, path: str | Path, iterator_class: Iterable) -> None:
        super().__init__()
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise RuntimeError(f'{path}: does not exist.')
        self.__path = path
        self.__iterator_class = iterator_class

    def __iter__(self) -> Iterable:
        return self.__iterator_class(self.__path)

    def __getitem__(self, index) -> NoReturn:
        raise NotImplementedError('Not implemented.')
