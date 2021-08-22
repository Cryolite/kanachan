#!/usr/bin/env python3

import json
import torch
import torch.nn.functional
from torch.utils.data import get_worker_info
from kanachan.constants import (
    NUM_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES, NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_POSITIONAL_FEATURES, MAX_LENGTH_OF_POSITIONAL_FEATURES,
    NUM_ACTIONS, MAX_NUM_ACTION_CANDIDATES)
from kanachan import common


class IteratorAdaptor(object):
    def __init__(self, path, num_dimensions, dtype) -> None:
        self.__fp = open(path)
        self.__num_dimensions = num_dimensions
        self.__dtype = dtype

        if get_worker_info() is not None:
            try:
                for i in range(get_worker_info().id):
                    next(self.__fp)
            except StopIteration as e:
                pass

    def __parse_line(self, line: str):
        line = line.rstrip('\n')
        uuid, sparse, numeric, positional, candidates, index, results = line.split('\t')

        sparse = json.loads('[' + sparse + ']')
        if len(sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
            raise RuntimeError(f'{uuid}: {len(sparse)}')
        for i in sparse:
            if i >= NUM_SPARSE_FEATURES:
                raise RuntimeError(f'{uuid}: {i}')
        for i in range(len(sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
            # padding
            sparse.append(NUM_SPARSE_FEATURES)
        sparse = torch.tensor(sparse, device='cpu', dtype=torch.int32)

        numeric = json.loads('[' + numeric + ']')
        if len(numeric) != NUM_NUMERIC_FEATURES:
            raise RuntimeError(uuid)
        numeric[2:] = [s / 10000.0 for s in numeric[2:]]
        numeric = torch.tensor(numeric, device='cpu', dtype=self.__dtype)
        numeric = torch.unsqueeze(numeric, 1)
        numeric = torch.nn.functional.pad(
            numeric, (0, self.__num_dimensions - 1))

        positional = json.loads('[' + positional + ']')
        if len(positional) > MAX_LENGTH_OF_POSITIONAL_FEATURES:
            raise RuntimeError(f'{uuid}: {len(positional)}')
        for p in positional:
            if p >= NUM_TYPES_OF_POSITIONAL_FEATURES:
                raise RuntimeError(f'{uuid}: {p}')
        for i in range(len(positional), MAX_LENGTH_OF_POSITIONAL_FEATURES):
            positional.append(NUM_TYPES_OF_POSITIONAL_FEATURES)
        positional = torch.tensor(positional, device='cpu', dtype=torch.int32)

        candidates = json.loads('[' + candidates + ']')
        if len(candidates) > MAX_NUM_ACTION_CANDIDATES:
            raise RuntimeError(f'{uuid}: {len(candidates)}')
        for a in candidates:
            if a >= NUM_ACTIONS:
                raise RuntimeError(f'{uuid}: {a}')
        for i in range(len(candidates), MAX_NUM_ACTION_CANDIDATES):
            candidates.append(NUM_ACTIONS)
        candidates = torch.tensor(candidates, device='cpu', dtype=torch.int32)

        index = torch.tensor(int(index), device='cpu', dtype=torch.int64)

        return sparse, numeric, positional, candidates, index

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
