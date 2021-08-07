#!/usr/bin/env python3

import json
import torch
import torch.nn.functional
from kanachan import common


class IteratorAdaptorBase(object):
    def __init__(self, fp, dimension, num_result_categories, result_index, device=None, dtype=None) -> None:
        self.__fp = fp
        self.__dimension = dimension
        self.__num_result_categories = num_result_categories
        self.__result_index = result_index
        self.__device = device
        self.__dtype = dtype

    def __next__(self):
        line = next(self.__fp)
        line = line.rstrip('\n')
        uuid, sparse, floats, action, results = line.split('\t')

        sparse_feature = json.loads('[' + sparse + ']')
        if len(sparse_feature) > common.MAX_NUM_ACTIVE_SPARSE_FEATURES:
            raise RuntimeError(uuid)
        for i in sparse_feature:
            if i >= common.NUM_SPARSE_FEATURES:
                raise RuntimeError(uuid)
        for i in range(len(sparse_feature), common.MAX_NUM_ACTIVE_SPARSE_FEATURES):
            sparse_feature.append(common.NUM_SPARSE_FEATURES)
        sparse_feature = torch.tensor(sparse_feature, device=self.__device, dtype=torch.int32)

        float_feature = json.loads('[' + floats + ']')
        if len(float_feature) != common.NUM_FLOAT_FEATURES:
            raise RuntimeError(uuid)
        float_feature = torch.tensor(float_feature, device=self.__device, dtype=self.__dtype)
        float_feature = torch.unsqueeze(float_feature, 1)
        float_feature = torch.nn.functional.pad(float_feature, (0, self.__dimension - 1))

        action = torch.tensor(int(action), device=self.__device, dtype=torch.int32)
        action = torch.unsqueeze(action, 0)

        results = json.loads('[' + results + ']')
        if self.__num_result_categories is None:
            # Regression
            y = results[self.__result_index]
            y = torch.tensor(y, device=self.__device, dtype=self.__dtype)
        else:
            y = torch.tensor(results[self.__result_index], device=self.__device, dtype=torch.int64)

        return (sparse_feature, float_feature, action, y)
