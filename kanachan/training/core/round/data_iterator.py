from pathlib import Path
import gzip
import bz2
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import get_worker_info
from kanachan.constants import (
    ROUND_NUM_TYPES_OF_SPARSE_FEATURES,
    ROUND_NUM_SPARSE_FEATURES,
    ROUND_NUM_NUMERIC_FEATURES,
    ROUND_NUM_RESULTS,
)


class DataIterator:
    def __init__(
        self, path: Path, num_skip_samples: int, local_rank: int
    ) -> None:
        if path.suffix == ".gz":
            self.__fp = gzip.open(path, mode="rt", encoding="UTF-8")
        elif path.suffix == ".bz2":
            self.__fp = bz2.open(path, mode="rt", encoding="UTF-8")
        else:
            self.__fp = open(path, encoding="UTF-8")

        worker_info = get_worker_info()

        if num_skip_samples > 0:
            is_primary_worker = worker_info is None or worker_info.id == 0
            with tqdm(
                desc="Skipping leading samples...",
                total=num_skip_samples,
                maxinterval=0.1,
                disable=(local_rank != 0 or not is_primary_worker),
                unit="samples",
                smoothing=0.0,
            ) as progress:
                for _ in range(num_skip_samples):
                    self.__fp.readline()
                    progress.update()

        if worker_info is not None:
            try:
                for _ in range(worker_info.id):
                    next(self.__fp)
            except StopIteration as _:
                pass

    def __del__(self) -> None:
        self.__fp.close()

    def __parse_line(self, line: str) -> tuple[Tensor, Tensor, Tensor]:
        line = line.rstrip("\n")
        uuid, sparse, numeric, result = line.split("\t")

        sparse = [int(x) for x in sparse.split(",")]
        if len(sparse) != ROUND_NUM_SPARSE_FEATURES:
            errmsg = f"{uuid}: {len(sparse)}"
            raise RuntimeError(errmsg)
        for x in sparse:
            if x >= ROUND_NUM_TYPES_OF_SPARSE_FEATURES:
                errmsg = f"{uuid}: {x}"
                raise RuntimeError(errmsg)
        sparse = torch.tensor(
            sparse, device=torch.device("cpu"), dtype=torch.int32
        )

        numeric = [int(x) for x in numeric.split(",")]
        if len(numeric) != ROUND_NUM_NUMERIC_FEATURES:
            raise RuntimeError(uuid)
        numeric = torch.tensor(
            numeric, device=torch.device("cpu"), dtype=torch.int32
        )

        result = [int(x) for x in result.split(",")]
        if len(result) != ROUND_NUM_RESULTS:
            raise RuntimeError(uuid)
        result = torch.tensor(
            result, device=torch.device("cpu"), dtype=torch.int32
        )

        return sparse, numeric, result

    def __next__(self) -> tuple[Tensor, Tensor, Tensor]:
        worker_info = get_worker_info()
        if worker_info is None:
            line = next(self.__fp)
            return self.__parse_line(line)
        else:
            line = next(self.__fp)
            try:
                assert worker_info.num_workers >= 1
                for _ in range(worker_info.num_workers - 1):
                    next(self.__fp)
            except StopIteration as _:
                pass
            return self.__parse_line(line)
