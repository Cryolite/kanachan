#!/usr/bin/env python3

import math
from typing import List
import torch
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES)


class TestModel(object):
    def __init__(self, rounds: List[dict]) -> None:
        self.__rounds = rounds
        self.__round_index = 0
        self.__decision_index = 0

    def __call__(self, features: List[List[int]]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(len(features))
        sparse = features[0]
        numeric = features[1]
        progression = features[2]
        candidates = features[3]

        if len(candidates) == 0:
            raise ValueError(f'''No candidate.
{sparse}
{numeric}
{progression}
{candidates}''')

        while self.__round_index < len(self.__rounds):
            _round = self.__rounds[self.__round_index]
            decisions = _round['decisions']
            if self.__decision_index >= len(decisions):
                self.__round_index += 1
                self.__decision_index = 0
                continue
            decision = decisions[self.__decision_index]
            error = False
            message = ''
            sparse = torch.squeeze(sparse)
            sparse_ = list(decision['sparse'])
            for _ in range(len(sparse_), MAX_NUM_ACTIVE_SPARSE_FEATURES):
                # Padding.
                sparse_.append(NUM_TYPES_OF_SPARSE_FEATURES)
            sparse_ = torch.tensor(
                sparse_, device=sparse.device, dtype=sparse.dtype)
            if torch.any(sparse != sparse_).item():
                error = True
                message += "sparse != decision['sparse']\n"
            numeric = torch.squeeze(numeric)
            numeric_ = list(decision['numeric'])
            numeric_[2:] = [float(x) / 10000.0 for x in numeric_[2:]]
            numeric_ = torch.tensor(
                numeric_, device=numeric.device, dtype=numeric.dtype)
            if torch.any(numeric != numeric_).item():
                error = True
                message += "numeric != decision['numeric']\n"
            progression = torch.squeeze(progression)
            progression_ = list(decision['progression'])
            for _ in range(len(progression_), MAX_LENGTH_OF_PROGRESSION_FEATURES):
                # Padding.
                progression_.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
            progression_ = torch.tensor(
                progression_, device=progression.device,
                dtype=progression.dtype)
            if torch.any(progression != progression_).item():
                error = True
                message += "progression != decision['progression']\n"
            candidates = torch.squeeze(candidates)
            candidates_ = list(decision['candidates'])
            for _ in range(len(candidates_), MAX_NUM_ACTION_CANDIDATES):
                # Padding.
                candidates_.append(NUM_TYPES_OF_ACTIONS + 1)
            candidates_ = torch.tensor(
                candidates_, device=candidates.device, dtype=candidates.dtype)
            if torch.any(candidates != candidates_).item():
                error = True
                message += "candidates != decision['candidates']\n"
            if error:
                message += f'''Decision error:
sparse: {sparse}
sparse: {sparse_}
numeric: {numeric}
numeric: {numeric_}
progression: {progression}
progression: {progression_}
candidates: {candidates}
candidates: {candidates_}
index: {decision["index"]}
'''
                if self.__decision_index >= 1:
                    prev_decision = decisions[self.__decision_index - 1]
                    message += f'''
===== Previous Decision =====
sparse: {prev_decision["sparse"]}
numeric: {prev_decision["numeric"]}
progression: {prev_decision["progression"]}
candidates: {prev_decision["candidates"]}
index: {prev_decision["index"]}
'''
                if self.__decision_index + 1 < len(decisions):
                    next_decision = decisions[self.__decision_index + 1]
                    message += f'''
===== Next Decision =====
sparse: {next_decision["sparse"]}
numeric: {next_decision["numeric"]}
progression: {next_decision["progression"]}
candidates: {next_decision["candidates"]}
index: {next_decision["index"]}
'''
                raise RuntimeError(message)
            index = decision['index']
            if index >= len(decision['candidates']):
                raise RuntimeError('Out of index.')
            self.__decision_index += 1
            prediction = torch.full(
                (MAX_NUM_ACTION_CANDIDATES,), -math.inf, device=numeric.device,
                dtype=numeric.dtype)
            prediction[index] = 0.0
            prediction = torch.unsqueeze(prediction, dim=0)
            return prediction

        raise RuntimeError('No more decision.')
