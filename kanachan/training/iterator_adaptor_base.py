#!/usr/bin/env python3

from pathlib import Path
import torch
import torch.nn.functional
from torch.utils.data import get_worker_info
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES, NUM_TYPES_OF_PROGRESSION_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES, NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES)


class IteratorAdaptorBase(object):
    def __init__(self, path: Path) -> None:
        self.__fp = open(path)

        if get_worker_info() is not None:
            try:
                for i in range(get_worker_info().id):
                    next(self.__fp)
            except StopIteration as e:
                pass

    def __del__(self) -> None:
        self.__fp.close()

    def __parse_line(self, line: str):
        line = line.rstrip('\n')
        uuid, sparse, numeric, progression, candidates, index, results = line.split('\t')

        sparse = [int(x) for x in sparse.split(',')]
        if len(sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
            raise RuntimeError(f'{uuid}: {len(sparse)}')
        for i in sparse:
            if i >= NUM_TYPES_OF_SPARSE_FEATURES:
                raise RuntimeError(f'{uuid}: {i}')
        for i in range(len(sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
            # padding
            sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
        sparse = torch.tensor(sparse, device='cpu', dtype=torch.int32)

        numeric = [int(x) for x in numeric.split(',')]
        if len(numeric) != NUM_NUMERIC_FEATURES:
            raise RuntimeError(uuid)
        numeric[2:] = [x / 10000.0 for x in numeric[2:]]
        numeric = torch.tensor(numeric, device='cpu', dtype=torch.float32)

        progression = [int(x) for x in progression.split(',')]
        if len(progression) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
            raise RuntimeError(f'{uuid}: {len(progression)}')
        for p in progression:
            if p >= NUM_TYPES_OF_PROGRESSION_FEATURES:
                raise RuntimeError(f'{uuid}: {p}')
        for i in range(len(progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
            # padding
            progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
        progression = torch.tensor(progression, device='cpu', dtype=torch.int32)

        candidates = [int(x) for x in candidates.split(',')]
        if len(candidates) > MAX_NUM_ACTION_CANDIDATES:
            raise RuntimeError(f'{uuid}: {len(candidates)}')
        for a in candidates:
            if a >= NUM_TYPES_OF_ACTIONS:
                raise RuntimeError(f'{uuid}: {a}')
        for i in range(len(candidates), MAX_NUM_ACTION_CANDIDATES):
            # padding
            candidates.append(NUM_TYPES_OF_ACTIONS + 1)
        candidates = torch.tensor(candidates, device='cpu', dtype=torch.int32)

        index = int(index)
        index = torch.tensor(index, device='cpu', dtype=torch.int64)

        results = [int(x) for x in results.split(',')]

        return (sparse, numeric, progression, candidates, index, results)

    def __next__(self):
        if get_worker_info() is None:
            line = next(self.__fp)
            return self.__parse_line(line)
        else:
            line = next(self.__fp)
            try:
                assert(get_worker_info().num_workers >= 1)
                for i in range(get_worker_info().num_workers - 1):
                    next(self.__fp)
            except StopIteration as e:
                pass
            return self.__parse_line(line)
