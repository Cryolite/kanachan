#!/usr/bin/env python3

import json
import torch
import torch.nn.functional
from kanachan import common


POST_ZIMO_DELTA_ROUND_SCORE_INDEX = 32986


class IteratorAdaptor(object):
    def __init__(self, fp, dimension, dtype) -> None:
        self.__fp = fp
        self.__dimension = dimension
        self.__dtype = dtype

    def __next__(self):
        line = next(self.__fp)
        line = line.rstrip('\n')
        uuid, sparse, floats, action, result = line.split('\t')

        sparse_feature = json.loads('[' + sparse + ']')
        if len(sparse_feature) > common.MAX_NUM_ACTIVE_COMMON_SPARSE_FEATURES:
            raise RuntimeError(uuid)
        for i in sparse_feature:
            if i >= common.NUM_COMMON_SPARSE_FEATURES:
                raise RuntimeError(uuid)
        for i in range(len(sparse_feature), common.MAX_NUM_ACTIVE_COMMON_SPARSE_FEATURES):
            sparse_feature.append(common.NUM_COMMON_SPARSE_FEATURES)
        sparse_feature = torch.tensor(sparse_feature, dtype=torch.int32)

        float_feature = json.loads('[' + floats + ']')
        if len(float_feature) != common.NUM_COMMON_FLOAT_FEATURES:
            raise RuntimeError(uuid)
        float_feature = torch.tensor(float_feature, dtype=self.__dtype)
        float_feature = torch.unsqueeze(float_feature, 1)
        float_feature = torch.nn.functional.pad(float_feature, (0, self.__dimension - 1))

        action = torch.tensor(int(action), dtype=torch.int32)
        action = torch.unsqueeze(action, 0)

        result = json.loads('[' + result + ']')
        y = torch.tensor(result[0], dtype=self.__dtype)

        return (sparse_feature, float_feature, action, y)
