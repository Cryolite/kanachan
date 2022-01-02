#!/usr/bin/env python3

from typing import List
import sys
from traceback import print_exc


class TestModel(object):
    def __init__(self, rounds: List[dict]) -> None:
        self.__rounds = rounds
        self.__round_index = 0
        self.__decision_index = 0

    def __call__(self, features: List[List[int]]) -> int:
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
            if sparse != decision['sparse']:
                error = True
            if numeric != decision['numeric']:
                error = True
            if progression != decision['progression']:
                error = True
            if candidates != decision['candidates']:
                error = True
            if error:
                message = f'''Decision error:
sparse: {sparse}
sparse: {decision["sparse"]}
numeric: {numeric}
numeric: {decision["numeric"]}
progression: {progression}
progression: {decision["progression"]}
candidates: {candidates}
candidates: {decision["candidates"]}
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
            return decision['candidates'][index]

        raise RuntimeError('No more decision.')
