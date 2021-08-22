#!/usr/bin/env python3

import io
import json
import torch
import torch.nn.functional
from torch.utils.data import get_worker_info
from kanachan import common


class IteratorAdaptor(object):
    def __init__(self, path, dimension, target_index, target_num_classes, dtype) -> None:
        self.__fp = open(path)
        self.__dimension = dimension
        self.__target_index = target_index
        self.__target_num_classes = target_num_classes
        self.__dtype = dtype

        if get_worker_info() is not None:
            try:
                for i in range(get_worker_info().id):
                    next(self.__fp)
            except StopIteration as e:
                pass

    def __parse_line(self, line: str):
        line = line.rstrip('\n')
        uuid, sparse, numeric, positional, action, results = line.split('\t')

        sparse = json.loads('[' + sparse + ']')
        if len(sparse) > common.MAX_NUM_ACTIVE_SPARSE_FEATURES:
            raise RuntimeError(f'{uuid}: {len(sparse)}')
        for i in sparse:
            if i >= common.NUM_SPARSE_FEATURES:
                raise RuntimeError(f'{uuid}: {i}')
        for i in range(len(sparse), common.MAX_NUM_ACTIVE_SPARSE_FEATURES):
            # padding
            sparse.append(common.NUM_SPARSE_FEATURES)
        sparse = torch.tensor(sparse, device='cpu', dtype=torch.int32)

        numeric = json.loads('[' + numeric + ']')
        if len(numeric) != common.NUM_NUMERIC_FEATURES:
            raise RuntimeError(uuid)
        numeric[2:] = [x / 10000.0 for x in numeric[2:]]
        numeric = torch.tensor(numeric, device='cpu', dtype=self.__dtype)
        numeric = torch.unsqueeze(numeric, 1)
        numeric = torch.nn.functional.pad(numeric, (0, self.__dimension - 1))

        positional = json.loads('[' + positional + ']')
        if len(positional) > common.MAX_LENGTH_OF_POSITIONAL_FEATURES:
            raise RuntimeError(f'{uuid}: {len(positional)}')
        for p in positional:
            if p >= common.NUM_TYPES_OF_POSITIONAL_FEATURES:
                raise RuntimeError(f'{uuid}: {p}')
        for i in range(len(positional), common.MAX_LENGTH_OF_POSITIONAL_FEATURES):
            positional.append(common.NUM_TYPES_OF_POSITIONAL_FEATURES)
        positional = torch.tensor(positional, device='cpu', dtype=torch.int32)

        action = torch.tensor(int(action), device='cpu', dtype=torch.int32)
        action = torch.unsqueeze(action, 0)

        results = json.loads('[' + results + ']')
        if self.__target_num_classes is None:
            # Regression
            y = results[self.__target_index]
            if self.__target_index in (1, 2, 3, 4, 10):
                y /= 10000.0
            elif self.__target_index == 11:
                y /= 100.0
            y = torch.tensor(y, device='cpu', dtype=self.__dtype)
        else:
            y = results[self.__target_index]
            if y >= self.__target_num_classes:
                raise RuntimeError(f'{uuid}: y = {y}')
            y = torch.tensor(y, device='cpu', dtype=torch.int64)

        return (sparse, numeric, positional, action), y

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
