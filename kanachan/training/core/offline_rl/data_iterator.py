from pathlib import Path
import gzip
import bz2

from tqdm import tqdm
import torch
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


class DataIterator:
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
            errmsg = (
                f"{num_skip_samples}: An invalid value for `num_skip_samples`."
            )
            raise ValueError(errmsg)
        if rewrite_rooms is not None and (
            rewrite_rooms < 0 or 4 < rewrite_rooms
        ):
            errmsg = f"{rewrite_rooms}: An invalid value for `rewrite_rooms`."
            raise ValueError(errmsg)
        if rewrite_grades is not None and (
            rewrite_grades < 0 or 15 < rewrite_grades
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
                unit=" lines",
                smoothing=0.0,
            ) as progress:
                for _ in range(num_skip_samples):
                    self.__fp.readline()
                    progress.update()

        if worker_info is not None:
            try:
                assert worker_info is not None
                for _ in range(worker_info.id):
                    next(self.__fp)
            except StopIteration as _:
                pass

    def __del__(self) -> None:
        self.__fp.close()

    def __parse_line(self, line: str):
        line = line.rstrip("\n")
        columns = line.split("\t")
        if len(columns) not in (8, 10, 12):
            errmsg = f"An invalid line: {line}"
            raise RuntimeError(errmsg)

        (
            _,
            sparse_str,
            numeric_str,
            progression_str,
            candidates_str,
            action_str,
        ) = columns[:6]

        _sparse = [int(x) for x in sparse_str.split(",")]
        if len(_sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
            errmsg = f"{len(_sparse)} > {MAX_NUM_ACTIVE_SPARSE_FEATURES}"
            raise RuntimeError(errmsg)
        for x in _sparse:
            if x >= NUM_TYPES_OF_SPARSE_FEATURES:
                errmsg = f"{x} >= {NUM_TYPES_OF_SPARSE_FEATURES}"
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
            errmsg = f"{len(_numeric)} != {NUM_NUMERIC_FEATURES}"
            raise RuntimeError(errmsg)
        numeric = torch.tensor(
            _numeric, device=torch.device("cpu"), dtype=torch.int32
        )

        _progression = [int(x) for x in progression_str.split(",")]
        if len(_progression) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
            errmsg = (
                f"{len(_progression)}"
                f" > {MAX_LENGTH_OF_PROGRESSION_FEATURES}"
            )
            raise RuntimeError(errmsg)
        for x in _progression:
            if x >= NUM_TYPES_OF_PROGRESSION_FEATURES:
                errmsg = f"{x} >= {NUM_TYPES_OF_PROGRESSION_FEATURES}"
                raise RuntimeError(errmsg)
        for _ in range(len(_progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
            # padding
            _progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
        progression = torch.tensor(
            _progression, device=torch.device("cpu"), dtype=torch.int32
        )

        _candidates = [int(x) for x in candidates_str.split(",")]
        if len(_candidates) > MAX_NUM_ACTION_CANDIDATES:
            errmsg = f"{len(_candidates)} >= {MAX_NUM_ACTION_CANDIDATES}"
            raise RuntimeError(errmsg)
        for x in _candidates:
            if x >= NUM_TYPES_OF_ACTIONS:
                errmsg = f"{x} >= {NUM_TYPES_OF_ACTIONS}"
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

        if len(columns) in (10, 12):
            # Not end-of-game.
            (
                next_sparse_str,
                next_numeric_str,
                next_progression_str,
                next_candidates_str,
            ) = columns[6:10]

            _next_sparse = [int(x) for x in next_sparse_str.split(",")]
            if len(_next_sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
                errmsg = (
                    f"{len(_next_sparse)}"
                    f" > {MAX_NUM_ACTIVE_SPARSE_FEATURES}"
                )
                raise RuntimeError(errmsg)
            for x in _next_sparse:
                if x >= NUM_TYPES_OF_SPARSE_FEATURES:
                    errmsg = f"{x} >= {NUM_TYPES_OF_SPARSE_FEATURES}"
                    raise RuntimeError(errmsg)
            for _ in range(len(_next_sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
                # padding
                _next_sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
            next_sparse = torch.tensor(
                _next_sparse, device=torch.device("cpu"), dtype=torch.int32
            )
            if self.__rewrite_rooms is not None:
                next_sparse[0] = self.__rewrite_rooms
            if self.__rewrite_grades is not None:
                next_sparse[2] = 7 + self.__rewrite_grades
                next_sparse[3] = 23 + self.__rewrite_grades
                next_sparse[4] = 39 + self.__rewrite_grades
                next_sparse[5] = 55 + self.__rewrite_grades

            _next_numeric = [int(x) for x in next_numeric_str.split(",")]
            if len(_next_numeric) != NUM_NUMERIC_FEATURES:
                errmsg = f"{len(_next_numeric)} != {NUM_NUMERIC_FEATURES}"
                raise RuntimeError(errmsg)
            next_numeric = torch.tensor(
                _next_numeric, device=torch.device("cpu"), dtype=torch.int32
            )

            _next_progression = [
                int(x) for x in next_progression_str.split(",")
            ]
            if len(_next_progression) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
                errmsg = (
                    f"{len(_next_progression)}"
                    f" > {MAX_LENGTH_OF_PROGRESSION_FEATURES}"
                )
                raise RuntimeError(errmsg)
            for x in _next_progression:
                if x >= NUM_TYPES_OF_PROGRESSION_FEATURES:
                    errmsg = f"{x} >= {NUM_TYPES_OF_PROGRESSION_FEATURES}"
                    raise RuntimeError(errmsg)
            for _ in range(
                len(_next_progression), MAX_LENGTH_OF_PROGRESSION_FEATURES
            ):
                # padding
                _next_progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
            next_progression = torch.tensor(
                _next_progression,
                device=torch.device("cpu"),
                dtype=torch.int32,
            )

            _next_candidates = [int(x) for x in next_candidates_str.split(",")]
            if len(_next_candidates) > MAX_NUM_ACTION_CANDIDATES:
                errmsg = (
                    f"{len(_next_candidates)}"
                    f" > {MAX_NUM_ACTION_CANDIDATES}"
                )
                raise RuntimeError(errmsg)
            for x in _next_candidates:
                if x >= NUM_TYPES_OF_ACTIONS:
                    errmsg = f"{x} >= {NUM_TYPES_OF_ACTIONS}"
                    raise RuntimeError(errmsg)
            for _ in range(len(_next_candidates), MAX_NUM_ACTION_CANDIDATES):
                # padding
                _next_candidates.append(NUM_TYPES_OF_ACTIONS)
            next_candidates = torch.tensor(
                _next_candidates, device=torch.device("cpu"), dtype=torch.int32
            )

            if len(columns) == 10:
                # Not end-of-game nor end-of-round.
                round_summary = torch.full(
                    (MAX_NUM_ROUND_SUMMARY,),
                    NUM_TYPES_OF_ROUND_SUMMARY,
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                )
                results = torch.zeros(
                    NUM_RESULTS,
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                )
                end_of_round = torch.tensor(
                    False, device=torch.device("cpu"), dtype=torch.bool
                )
            else:
                # End-of-round but not End-of-game
                assert len(columns) == 12
                round_summary_str, results_str = columns[10:]

                _round_summary = [int(x) for x in round_summary_str.split(",")]
                if len(_round_summary) == 0:
                    errmsg = f"An invalid line: {line}"
                    raise RuntimeError(errmsg)
                if len(_round_summary) > MAX_NUM_ROUND_SUMMARY:
                    errmsg = f"{len(_round_summary)} > {MAX_NUM_ROUND_SUMMARY}"
                    raise RuntimeError(errmsg)
                for x in _round_summary:
                    if x >= NUM_TYPES_OF_ROUND_SUMMARY:
                        errmsg = f"{x} >= {NUM_TYPES_OF_ROUND_SUMMARY}"
                        raise RuntimeError(errmsg)
                for _ in range(len(_round_summary), MAX_NUM_ROUND_SUMMARY):
                    # Padding
                    _round_summary.append(NUM_TYPES_OF_ROUND_SUMMARY)
                round_summary = torch.tensor(
                    _round_summary,
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                )

                _results = [int(x) for x in results_str.split(",")]
                if len(_results) != NUM_RESULTS - 4:
                    errmsg = f"{len(_results)} != {NUM_RESULTS - 4}"
                    raise RuntimeError(errmsg)
                _results.extend([0, 0, 0, 0])
                results = torch.tensor(
                    _results, device=torch.device("cpu"), dtype=torch.int32
                )

                end_of_round = torch.tensor(
                    True, device=torch.device("cpu"), dtype=torch.bool
                )

            done = torch.tensor(
                False, device=torch.device("cpu"), dtype=torch.bool
            )

            return (
                sparse,
                numeric,
                progression,
                candidates,
                action,
                next_sparse,
                next_numeric,
                next_progression,
                next_candidates,
                round_summary,
                results,
                end_of_round,
                done,
            )

        # End-of-game
        assert len(columns) == 8

        _dummy_sparse = [
            NUM_TYPES_OF_SPARSE_FEATURES
        ] * MAX_NUM_ACTIVE_SPARSE_FEATURES
        dummy_sparse = torch.tensor(
            _dummy_sparse, device=torch.device("cpu"), dtype=torch.int32
        )

        dummy_numeric = torch.zeros(
            NUM_NUMERIC_FEATURES, device=torch.device("cpu"), dtype=torch.int32
        )

        _dummy_progression = [
            NUM_TYPES_OF_PROGRESSION_FEATURES
        ] * MAX_LENGTH_OF_PROGRESSION_FEATURES
        dummy_progression = torch.tensor(
            _dummy_progression, device=torch.device("cpu"), dtype=torch.int32
        )

        _dummy_candidates = [NUM_TYPES_OF_ACTIONS] * MAX_NUM_ACTION_CANDIDATES
        dummy_candidates = torch.tensor(
            _dummy_candidates, device=torch.device("cpu"), dtype=torch.int32
        )

        round_summary_str, results_str = columns[6:]

        _round_summary = [int(x) for x in round_summary_str.split(",")]
        if len(_round_summary) == 0:
            errmsg = f"An invalid line: {line}"
            raise RuntimeError(errmsg)
        if len(_round_summary) > MAX_NUM_ROUND_SUMMARY:
            errmsg = f"{len(_round_summary)} > {MAX_NUM_ROUND_SUMMARY}"
            raise RuntimeError(errmsg)
        for x in _round_summary:
            if x >= NUM_TYPES_OF_ROUND_SUMMARY:
                errmsg = f"{x} >= {NUM_TYPES_OF_ROUND_SUMMARY}"
                raise RuntimeError(errmsg)
        for _ in range(len(_round_summary), MAX_NUM_ROUND_SUMMARY):
            # Padding
            _round_summary.append(NUM_TYPES_OF_ROUND_SUMMARY)
        round_summary = torch.tensor(
            _round_summary, device=torch.device("cpu"), dtype=torch.int32
        )

        _results = [int(x) for x in results_str.split(",")]
        if len(_results) != NUM_RESULTS:
            errmsg = f"{len(_results)} != {NUM_RESULTS}"
            raise RuntimeError(errmsg)
        results = torch.tensor(
            _results, device=torch.device("cpu"), dtype=torch.int32
        )

        end_of_round = torch.tensor(
            True, device=torch.device("cpu"), dtype=torch.bool
        )

        done = torch.tensor(True, device=torch.device("cpu"), dtype=torch.bool)

        return (
            sparse,
            numeric,
            progression,
            candidates,
            action,
            dummy_sparse,
            dummy_numeric,
            dummy_progression,
            dummy_candidates,
            round_summary,
            results,
            end_of_round,
            done,
        )

    def __next__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            line = next(self.__fp)
            return self.__parse_line(line)
        else:
            assert worker_info.num_workers >= 1
            line = next(self.__fp)
            try:
                for _ in range(worker_info.num_workers - 1):
                    next(self.__fp)
            except StopIteration:
                pass
            return self.__parse_line(line)
