from pathlib import Path
import gzip
import bz2
import torch
from torch.utils.data import get_worker_info
from kanachan.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES, NUM_TYPES_OF_PROGRESSION_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES, NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES,
    NUM_TYPES_OF_ROUND_SUMMARY, MAX_NUM_ROUND_SUMMARY, RL_NUM_RESULTS
)


class DataIterator:
    def __init__(self, path: Path) -> None:
        if path.suffix == '.gz':
            self.__fp = gzip.open(path, mode='rt', encoding='UTF-8')
        elif path.suffix == '.bz2':
            self.__fp = bz2.open(path, mode='rt', encoding='UTF-8')
        else:
            self.__fp = open(path, encoding='UTF-8')

        if get_worker_info() is not None:
            try:
                for _ in range(get_worker_info().id):
                    next(self.__fp)
            except StopIteration as _:
                pass

    def __del__(self) -> None:
        self.__fp.close()

    def __parse_line(self, line: str):
        line = line.rstrip('\n')
        columns = line.split('\t')
        if len(columns) not in (8, 10, 12):
            raise RuntimeError(f'An invalid line: {line}')

        _, sparse, numeric, progression, candidates, action = columns[:6]

        sparse = [int(x) for x in sparse.split(',')]
        if len(sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
            raise RuntimeError(f'{len(sparse)} > {MAX_NUM_ACTIVE_SPARSE_FEATURES}')
        for x in sparse:
            if x >= NUM_TYPES_OF_SPARSE_FEATURES:
                raise RuntimeError(f'{x} >= {NUM_TYPES_OF_SPARSE_FEATURES}')
        for _ in range(len(sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
            # padding
            sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
        sparse = torch.tensor(sparse, device=torch.device('cpu'), dtype=torch.int32)

        numeric = [int(x) for x in numeric.split(',')]
        if len(numeric) != NUM_NUMERIC_FEATURES:
            raise RuntimeError(f'{len(numeric)} != {NUM_NUMERIC_FEATURES}')
        numeric = torch.tensor(numeric, device=torch.device('cpu'), dtype=torch.int32)

        progression = [int(x) for x in progression.split(',')]
        if len(progression) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
            raise RuntimeError(f'{len(progression)} > {MAX_LENGTH_OF_PROGRESSION_FEATURES}')
        for x in progression:
            if x >= NUM_TYPES_OF_PROGRESSION_FEATURES:
                raise RuntimeError(f'{x} >= {NUM_TYPES_OF_PROGRESSION_FEATURES}')
        for _ in range(len(progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
            # padding
            progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
        progression = torch.tensor(progression, device=torch.device('cpu'), dtype=torch.int32)

        candidates = [int(x) for x in candidates.split(',')]
        if len(candidates) > MAX_NUM_ACTION_CANDIDATES:
            raise RuntimeError(f'{len(candidates)} >= {MAX_NUM_ACTION_CANDIDATES}')
        for x in candidates:
            if x >= NUM_TYPES_OF_ACTIONS:
                raise RuntimeError(f'{x} >= {NUM_TYPES_OF_ACTIONS}')
        for _ in range(len(candidates), MAX_NUM_ACTION_CANDIDATES):
            # padding
            candidates.append(NUM_TYPES_OF_ACTIONS)
        candidates = torch.tensor(candidates, device=torch.device('cpu'), dtype=torch.int32)

        action = int(action)
        action = torch.tensor(action, device=torch.device('cpu'), dtype=torch.int32)

        if len(columns) in (10, 12):
            # Not end-of-game.
            next_sparse, next_numeric, next_progression, next_candidates = columns[6:10]

            next_sparse = [int(x) for x in next_sparse.split(',')]
            if len(next_sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
                raise RuntimeError(f'{len(next_sparse)} > {MAX_NUM_ACTIVE_SPARSE_FEATURES}')
            for x in next_sparse:
                if x >= NUM_TYPES_OF_SPARSE_FEATURES:
                    raise RuntimeError(f'{x} >= {NUM_TYPES_OF_SPARSE_FEATURES}')
            for _ in range(len(next_sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
                # padding
                next_sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
            next_sparse = torch.tensor(
                next_sparse, device=torch.device('cpu'), dtype=torch.int32)

            next_numeric = [int(x) for x in next_numeric.split(',')]
            if len(next_numeric) != NUM_NUMERIC_FEATURES:
                raise RuntimeError(f'{len(next_numeric)} != {NUM_NUMERIC_FEATURES}')
            next_numeric = torch.tensor(
                next_numeric, device=torch.device('cpu'), dtype=torch.int32)

            next_progression = [int(x) for x in next_progression.split(',')]
            if len(next_progression) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
                raise RuntimeError(
                    f'{len(next_progression)} > {MAX_LENGTH_OF_PROGRESSION_FEATURES}')
            for x in next_progression:
                if x >= NUM_TYPES_OF_PROGRESSION_FEATURES:
                    raise RuntimeError(f'{x} >= {NUM_TYPES_OF_PROGRESSION_FEATURES}')
            for _ in range(len(next_progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
                # padding
                next_progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
            next_progression = torch.tensor(
                next_progression, device=torch.device('cpu'), dtype=torch.int32)

            next_candidates = [int(x) for x in next_candidates.split(',')]
            if len(next_candidates) > MAX_NUM_ACTION_CANDIDATES:
                raise RuntimeError(f'{len(next_candidates)} > {MAX_NUM_ACTION_CANDIDATES}')
            for x in next_candidates:
                if x >= NUM_TYPES_OF_ACTIONS:
                    raise RuntimeError(f'{x} >= {NUM_TYPES_OF_ACTIONS}')
            for _ in range(len(next_candidates), MAX_NUM_ACTION_CANDIDATES):
                # padding
                next_candidates.append(NUM_TYPES_OF_ACTIONS)
            next_candidates = torch.tensor(
                next_candidates, device=torch.device('cpu'), dtype=torch.int32)

            if len(columns) == 10:
                # Not end-of-game nor end-of-round.
                round_summary = torch.full(
                    (MAX_NUM_ROUND_SUMMARY,), NUM_TYPES_OF_ROUND_SUMMARY,
                    device=torch.device('cpu'), dtype=torch.int32)
                results = torch.zeros(
                    RL_NUM_RESULTS, device=torch.device('cpu'), dtype=torch.int32)
                beginning_of_round = torch.tensor(
                    False, device=torch.device('cpu'), dtype=torch.bool)
            else:
                # End-of-round but not End-of-game
                assert len(columns) == 12
                round_summary, results = columns[10:]

                round_summary = [int(x) for x in round_summary.split(',')]
                if len(round_summary) == 0:
                    raise RuntimeError(f'An invalid line: {line}')
                if len(round_summary) > MAX_NUM_ROUND_SUMMARY:
                    raise RuntimeError(f'{len(round_summary)} > {MAX_NUM_ROUND_SUMMARY}')
                for x in round_summary:
                    if x >= NUM_TYPES_OF_ROUND_SUMMARY:
                        raise RuntimeError(f'{x} >= {NUM_TYPES_OF_ROUND_SUMMARY}')
                for _ in range(len(round_summary), MAX_NUM_ROUND_SUMMARY):
                    # Padding
                    round_summary.append(NUM_TYPES_OF_ROUND_SUMMARY)
                round_summary = torch.tensor(
                    round_summary, device=torch.device('cpu'), dtype=torch.int32)

                results = [int(x) for x in results.split(',')]
                if len(results) != RL_NUM_RESULTS:
                    raise RuntimeError(f'{len(results)} != {RL_NUM_RESULTS}')
                results = torch.tensor(results, device=torch.device('cpu'), dtype=torch.int32)

                beginning_of_round = torch.tensor(True, device=torch.device('cpu'), dtype=torch.bool)

            done = torch.tensor(False, device=torch.device('cpu'), dtype=torch.bool)

            return (
                sparse, numeric, progression, candidates, action,
                next_sparse, next_numeric, next_progression, next_candidates,
                round_summary, results, beginning_of_round, done)

        # End-of-game
        assert len(columns) == 8

        dummy_sparse = [NUM_TYPES_OF_SPARSE_FEATURES] * MAX_NUM_ACTIVE_SPARSE_FEATURES
        dummy_sparse = torch.tensor(dummy_sparse, device=torch.device('cpu'), dtype=torch.int32)

        dummy_numeric = torch.zeros(
            NUM_NUMERIC_FEATURES, device=torch.device('cpu'), dtype=torch.int32)

        dummy_progression = [NUM_TYPES_OF_PROGRESSION_FEATURES] * MAX_LENGTH_OF_PROGRESSION_FEATURES
        dummy_progression = torch.tensor(
            dummy_progression, device=torch.device('cpu'), dtype=torch.int32)

        dummy_candidates = [NUM_TYPES_OF_ACTIONS] * MAX_NUM_ACTION_CANDIDATES
        dummy_candidates = torch.tensor(
            dummy_candidates, device=torch.device('cpu'), dtype=torch.int32)

        round_summary, results = columns[6:]

        round_summary = [int(x) for x in round_summary.split(',')]
        if len(round_summary) == 0:
            raise RuntimeError(f'An invalid line: {line}')
        if len(round_summary) > MAX_NUM_ROUND_SUMMARY:
            raise RuntimeError(f'{len(round_summary)} > {MAX_NUM_ROUND_SUMMARY}')
        for x in round_summary:
            if x >= NUM_TYPES_OF_ROUND_SUMMARY:
                raise RuntimeError(f'{x} >= {NUM_TYPES_OF_ROUND_SUMMARY}')
        for _ in range(len(round_summary), MAX_NUM_ROUND_SUMMARY):
            # Padding
            round_summary.append(NUM_TYPES_OF_ROUND_SUMMARY)
        round_summary = torch.tensor(round_summary, device=torch.device('cpu'), dtype=torch.int32)

        results = [int(x) for x in results.split(',')]
        if len(results) != RL_NUM_RESULTS:
            raise RuntimeError(f'{len(results)} != {RL_NUM_RESULTS}')
        results = torch.tensor(results, device=torch.device('cpu'), dtype=torch.int32)

        beginning_of_round = torch.tensor(False, device=torch.device('cpu'), dtype=torch.bool)

        done = torch.tensor(True, device=torch.device('cpu'), dtype=torch.bool)

        return (
            sparse, numeric, progression, candidates, action,
            dummy_sparse, dummy_numeric, dummy_progression, dummy_candidates,
            round_summary, results, beginning_of_round, done)

    def __next__(self):
        if get_worker_info() is None:
            line = next(self.__fp)
            return self.__parse_line(line)
        else:
            line = next(self.__fp)
            try:
                assert get_worker_info().num_workers >= 1
                for _ in range(get_worker_info().num_workers - 1):
                    next(self.__fp)
            except StopIteration as _:
                pass
            return self.__parse_line(line)
