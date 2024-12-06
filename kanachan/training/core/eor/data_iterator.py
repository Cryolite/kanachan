from pathlib import Path
import gzip
import bz2
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import get_worker_info
from kanachan.constants import (
    EOR_NUM_TYPES_OF_SPARSE_FEATURES,
    EOR_NUM_SPARSE_FEATURES,
    EOR_NUM_NUMERIC_FEATURES,
    EOR_NUM_GAME_RESULT,
)


class DataIterator:
    def __init__(
        self,
        path: Path,
        num_skip_samples: int,
        rewrite_rooms: int | None,
        rewrite_grades: int | None,
        local_rank: int,
    ) -> None:
        if num_skip_samples < 0:
            errmsg = (
                f"{num_skip_samples}: An invalid value for `num_skip_samples`."
            )
            raise ValueError(errmsg)
        if rewrite_rooms is not None and (
            rewrite_rooms < 0 or rewrite_rooms > 4
        ):
            errmsg = f"{rewrite_rooms}: An invalid value for `rewrite_rooms`."
            raise ValueError(errmsg)
        if rewrite_grades is not None and (
            rewrite_grades < 0 or rewrite_grades > 14
        ):
            errmsg = (
                f"{rewrite_grades}: An invalid value for `rewrite_grades`."
            )
            raise ValueError(errmsg)

        if path.suffix == ".gz":
            self.__fp = gzip.open(path, mode="rt", encoding="UTF-8")
        elif path.suffix == ".bz2":
            self.__fp = bz2.open(path, mode="rt", encoding="UTF-8")
        else:
            self.__fp = open(path, encoding="UTF-8")

        self.__rewrite_rooms = rewrite_rooms
        self.__rewrite_grades = rewrite_grades

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
        uuid, sparse_str, numeric_str, game_result_str = line.split("\t")

        _sparse = [int(x) for x in sparse_str.split(",")]
        if len(_sparse) != EOR_NUM_SPARSE_FEATURES:
            errmsg = f"{uuid}: {len(_sparse)}"
            raise RuntimeError(errmsg)
        for x in _sparse:
            if x >= EOR_NUM_TYPES_OF_SPARSE_FEATURES:
                errmsg = f"{uuid}: {x}"
                raise RuntimeError(errmsg)
        sparse = torch.tensor(_sparse, device="cpu", dtype=torch.int32)
        if self.__rewrite_rooms is not None:
            sparse[0] = self.__rewrite_rooms
        if self.__rewrite_grades is not None:
            sparse[2] = 7 + self.__rewrite_grades
            sparse[3] = 23 + self.__rewrite_grades
            sparse[4] = 39 + self.__rewrite_grades
            sparse[5] = 55 + self.__rewrite_grades

        _numeric = [int(x) for x in numeric_str.split(",")]
        if len(_numeric) != EOR_NUM_NUMERIC_FEATURES:
            raise RuntimeError(uuid)
        numeric = torch.tensor(_numeric, device="cpu", dtype=torch.int32)

        _game_result = [int(x) for x in game_result_str.split(",")]
        if len(_game_result) != EOR_NUM_GAME_RESULT:
            raise RuntimeError(uuid)
        game_result = torch.tensor(
            _game_result, device="cpu", dtype=torch.int32
        )

        return sparse, numeric, game_result

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
