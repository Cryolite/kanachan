from pathlib import Path
import gzip
import bz2
import torch
from torch import Tensor
from torch.utils.data import get_worker_info
from kanachan.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES, NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES, NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ROUND_SUMMARY, MAX_NUM_ROUND_SUMMARY, NUM_RESULTS)


_Result = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class DataIterator(object):
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

    def __parse_line(self, line: str) -> _Result:
        line = line.rstrip('\n')
        uuid, sparse, numeric, progression, candidates, action, round_summary, results = line.split('\t')

        sparse = [int(x) for x in sparse.split(',')]
        if len(sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
            raise RuntimeError(f'{uuid}: {len(sparse)}')
        for x in sparse:
            if x >= NUM_TYPES_OF_SPARSE_FEATURES:
                raise RuntimeError(f'{uuid}: {x}')
        for _ in range(len(sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
            # padding
            sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
        sparse = torch.tensor(sparse, device=torch.device('cpu'), dtype=torch.int32)

        numeric = [int(x) for x in numeric.split(',')]
        if len(numeric) != NUM_NUMERIC_FEATURES:
            raise RuntimeError(uuid)
        numeric = torch.tensor(numeric, device=torch.device('cpu'), dtype=torch.int32)

        progression = [int(x) for x in progression.split(',')]
        if len(progression) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
            raise RuntimeError(f'{uuid}: {len(progression)}')
        for x in progression:
            if x >= NUM_TYPES_OF_PROGRESSION_FEATURES:
                raise RuntimeError(f'{uuid}: {x}')
        for _ in range(len(progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
            # padding
            progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
        progression = torch.tensor(progression, device=torch.device('cpu'), dtype=torch.int32)

        candidates = [int(x) for x in candidates.split(',')]
        if len(candidates) > MAX_NUM_ACTION_CANDIDATES:
            raise RuntimeError(f'{uuid}: {len(candidates)}')
        for x in candidates:
            if x >= NUM_TYPES_OF_ACTIONS:
                raise RuntimeError(f'{uuid}: {x}')
        for _ in range(len(candidates), MAX_NUM_ACTION_CANDIDATES):
            # padding
            candidates.append(NUM_TYPES_OF_ACTIONS)
        candidates = torch.tensor(candidates, device=torch.device('cpu'), dtype=torch.int32)

        action = int(action)
        action = torch.tensor(action, device=torch.device('cpu'), dtype=torch.int32)

        round_summary = [int(x) for x in round_summary.split(',')]
        if len(round_summary) > MAX_NUM_ROUND_SUMMARY:
            raise RuntimeError(f'{uuid}: {len(round_summary)}')
        for x in round_summary:
            if x >= NUM_TYPES_OF_ROUND_SUMMARY:
                raise RuntimeError(f'{uuid}: {x}')
        for _ in range(len(round_summary), MAX_NUM_ROUND_SUMMARY):
            #padding
            round_summary.append(NUM_TYPES_OF_ROUND_SUMMARY)
        round_summary = torch.tensor(round_summary, device=torch.device('cpu'), dtype=torch.int32)

        results = [int(x) for x in results.split(',')]
        if len(results) != NUM_RESULTS:
            raise RuntimeError(f'{uuid}: {len(results)}')
        results = torch.tensor(results, device=torch.device('cpu'), dtype=torch.int32)

        return sparse, numeric, progression, candidates, action, round_summary, results

    def __next__(self) -> _Result:
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
