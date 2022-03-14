#!/usr/bin/env python3

from pathlib import Path
import gzip
import bz2
import torch
from torch.utils.data import get_worker_info
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES, NUM_TYPES_OF_PROGRESSION_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES, NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES)


class IteratorAdaptor(object):
    def __init__(self, path: Path) -> None:
        if path.suffix == '.gz':
            self.__fp = gzip.open(path, mode='rt', encoding='UTF-8')
        elif path.suffix == '.bz2':
            self.__fp = bz2.open(path, mode='rt', encoding='UTF-8')
        else:
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
        columns = line.split('\t')

        sparse, numeric, progression, candidates, index = columns[:5]

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
        if len(candidates) + 1 > MAX_NUM_ACTION_CANDIDATES:
            raise RuntimeError(f'{uuid}: {len(candidates)}')
        for a in candidates:
            if a >= NUM_TYPES_OF_ACTIONS:
                raise RuntimeError(f'{uuid}: {a}')
        # <VALUE>
        candidates.append(NUM_TYPES_OF_ACTIONS)
        for i in range(len(candidates), MAX_NUM_ACTION_CANDIDATES):
            # padding
            candidates.append(NUM_TYPES_OF_ACTIONS + 1)
        candidates = torch.tensor(candidates, device='cpu', dtype=torch.int32)

        index = int(index)
        index = torch.tensor(index, device='cpu', dtype=torch.int64)

        if len(columns) == 9:
            next_sparse, next_numeric, next_progression, next_candidates = columns[5:]

            next_sparse = [int(x) for x in next_sparse.split(',')]
            if len(next_sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
                raise RuntimeError(f'{uuid}: {len(next_sparse)}')
            for i in next_sparse:
                if i >= NUM_TYPES_OF_SPARSE_FEATURES:
                    raise RuntimeError(f'{uuid}: {i}')
            for i in range(len(next_sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
                # padding
                next_sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
            next_sparse = torch.tensor(
                next_sparse, device='cpu', dtype=torch.int32)

            next_numeric = [int(x) for x in next_numeric.split(',')]
            if len(next_numeric) != NUM_NUMERIC_FEATURES:
                raise RuntimeError(uuid)
            next_numeric[2:] = [x / 10000.0 for x in next_numeric[2:]]
            next_numeric = torch.tensor(
                next_numeric, device='cpu', dtype=torch.float32)

            next_progression = [int(x) for x in next_progression.split(',')]
            if len(next_progression) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
                raise RuntimeError(f'{uuid}: {len(next_progression)}')
            for p in next_progression:
                if p >= NUM_TYPES_OF_PROGRESSION_FEATURES:
                    raise RuntimeError(f'{uuid}: {p}')
            for i in range(len(next_progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
                # padding
                next_progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
            next_progression = torch.tensor(
                next_progression, device='cpu', dtype=torch.int32)

            next_candidates = [int(x) for x in next_candidates.split(',')]
            if len(next_candidates) + 1 > MAX_NUM_ACTION_CANDIDATES:
                raise RuntimeError(f'{uuid}: {len(next_candidates)}')
            for a in next_candidates:
                if a >= NUM_TYPES_OF_ACTIONS:
                    raise RuntimeError(f'{uuid}: {a}')
            # <VALUE>
            next_candidates.append(NUM_TYPES_OF_ACTIONS)
            for i in range(len(next_candidates), MAX_NUM_ACTION_CANDIDATES):
                # padding
                next_candidates.append(NUM_TYPES_OF_ACTIONS + 1)
            next_candidates = torch.tensor(
                next_candidates, device='cpu', dtype=torch.int32)

            reward = torch.tensor(0.0, device='cpu', dtype=torch.float32)

            return (
                sparse, numeric, progression, candidates, index,
                next_sparse, next_numeric, next_progression, next_candidates,
                reward)
        elif len(columns) == 6:
            reward = int(columns[5])
            reward /= 100.0
            reward = torch.tensor(reward, device='cpu', dtype=torch.float32)

            dummy_sparse = [NUM_TYPES_OF_SPARSE_FEATURES] * MAX_NUM_ACTIVE_SPARSE_FEATURES
            dummy_sparse = torch.tensor(dummy_sparse, device='cpu', dtype=torch.int32)

            dummy_numeric = [0, 0, 0, 0, 0, 0]
            dummy_numeric = torch.tensor(
                dummy_numeric, device='cpu', dtype=torch.float32)

            dummy_progression = [NUM_TYPES_OF_PROGRESSION_FEATURES] * MAX_LENGTH_OF_PROGRESSION_FEATURES
            dummy_progression = torch.tensor(
                dummy_progression, device='cpu', dtype=torch.int32)

            dummy_candidates = [NUM_TYPES_OF_ACTIONS]
            for i in range(len(dummy_candidates), MAX_NUM_ACTION_CANDIDATES):
                # padding
                dummy_candidates.append(NUM_TYPES_OF_ACTIONS + 1)
            dummy_candidates = torch.tensor(
                dummy_candidates, device='cpu', dtype=torch.int32)

            return (
                sparse, numeric, progression, candidates, index, dummy_sparse,
                dummy_numeric, dummy_progression, dummy_candidates, reward)

        raise RuntimeError(f'An invalid line: {line}')

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
