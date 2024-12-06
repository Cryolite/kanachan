from pathlib import Path
import gzip
import bz2
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import get_worker_info
from kanachan.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES,
    NUM_TYPES_OF_ROUND_SUMMARY,
    MAX_NUM_ROUND_SUMMARY,
    NUM_RESULTS,
)


_Result = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class DataIterator(object):
    def __init__(
        self,
        *,
        path: Path,
        num_skip_samples: int,
        rewrite_rooms: int | None,
        rewrite_grades: int | None,
        local_rank: int,
    ) -> None:
        if num_skip_samples < 0:
            errmsg = f"{num_skip_samples}: An invalid value for `num_skip_samples`."
            raise ValueError(errmsg)
        if rewrite_rooms is not None and (rewrite_rooms < 0 or 4 < rewrite_rooms):
            errmsg = f"{rewrite_rooms}: An invalid value for `rewrite_rooms`."
            raise ValueError(errmsg)
        if rewrite_grades is not None and (rewrite_grades < 0 or 15 < rewrite_grades):
            errmsg = f"{rewrite_grades}: An invalid value for `rewrite_grades`."
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

    def __parse_line(self, line: str) -> _Result:
        line = line.rstrip("\n")
        (
            uuid,
            sparse_str,
            numeric_str,
            progression_str,
            candidates_str,
            action_str,
            round_summary_str,
            results_str,
        ) = line.split("\t")

        _sparse = [int(x) for x in sparse_str.split(",")]
        if len(_sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
            errmsg = f"{uuid}: {len(_sparse)}"
            raise RuntimeError(errmsg)
        for x in _sparse:
            if x >= NUM_TYPES_OF_SPARSE_FEATURES:
                errmsg = f"{uuid}: {x}"
                raise RuntimeError(errmsg)
        for _ in range(len(_sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
            # padding
            _sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
        sparse = torch.tensor(
            _sparse, device=torch.device("cpu"), dtype=torch.int32
        )
        if self.__rewrite_rooms is not None:
            sparse[0] = self.__rewrite_rooms
        if self.__rewrite_grades is not None:
            sparse[2] = 7 + self.__rewrite_grades
            sparse[3] = 23 + self.__rewrite_grades
            sparse[4] = 39 + self.__rewrite_grades
            sparse[5] = 55 + self.__rewrite_grades

        _numeric = [int(x) for x in numeric_str.split(",")]
        if len(_numeric) != NUM_NUMERIC_FEATURES:
            raise RuntimeError(uuid)
        numeric = torch.tensor(
            _numeric, device=torch.device("cpu"), dtype=torch.int32
        )

        _progression = [int(x) for x in progression_str.split(",")]
        if len(_progression) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
            errmsg = f"{uuid}: {len(_progression)}"
            raise RuntimeError(errmsg)
        for x in _progression:
            if x >= NUM_TYPES_OF_PROGRESSION_FEATURES:
                errmsg = f"{uuid}: {x}"
                raise RuntimeError(errmsg)
        for _ in range(len(_progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
            # padding
            _progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
        progression = torch.tensor(
            _progression, device=torch.device("cpu"), dtype=torch.int32
        )

        _candidates = [int(x) for x in candidates_str.split(",")]
        if len(_candidates) > MAX_NUM_ACTION_CANDIDATES:
            errmsg = f"{uuid}: {len(_candidates)}"
            raise RuntimeError(errmsg)
        for x in _candidates:
            if x >= NUM_TYPES_OF_ACTIONS:
                errmsg = f"{uuid}: {x}"
                raise RuntimeError(errmsg)
        for _ in range(len(_candidates), MAX_NUM_ACTION_CANDIDATES):
            # padding
            _candidates.append(NUM_TYPES_OF_ACTIONS)
        candidates = torch.tensor(
            _candidates, device=torch.device("cpu"), dtype=torch.int32
        )

        _action = int(action_str)
        action = torch.tensor(
            _action, device=torch.device("cpu"), dtype=torch.int32
        )

        _round_summary = [int(x) for x in round_summary_str.split(",")]
        if len(_round_summary) > MAX_NUM_ROUND_SUMMARY:
            errmsg = f"{uuid}: {len(_round_summary)}"
            raise RuntimeError(errmsg)
        for x in _round_summary:
            if x >= NUM_TYPES_OF_ROUND_SUMMARY:
                errmsg = f"{uuid}: {x}"
                raise RuntimeError(errmsg)
        for _ in range(len(_round_summary), MAX_NUM_ROUND_SUMMARY):
            # padding
            _round_summary.append(NUM_TYPES_OF_ROUND_SUMMARY)
        round_summary = torch.tensor(
            _round_summary, device=torch.device("cpu"), dtype=torch.int32
        )

        _results = [int(x) for x in results_str.split(",")]
        if len(_results) != NUM_RESULTS:
            errmsg = f"{uuid}: {len(_results)}"
            raise RuntimeError(errmsg)
        results = torch.tensor(
            _results, device=torch.device("cpu"), dtype=torch.int32
        )

        return (
            sparse,
            numeric,
            progression,
            candidates,
            action,
            round_summary,
            results,
        )

    def __next__(self) -> _Result:
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
