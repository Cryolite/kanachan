#!/usr/bin/env python3

from pathlib import Path
from typing import (List, Union,)
from ._xiangting_calculator import XiangtingCalculator as Impl


_CONVERTER = [
     4,  0,  1,  2,  3,  4,  5,  6,  7,  8,
    13,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    22, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33
]


class XiangtingCalculator:
    def __init__(self, prefix: Union[str, Path]) -> None:
        if isinstance(prefix, Path):
            prefix = str(prefix)
        self.__impl = Impl(prefix)

    def calculate(self, hand: List[int], n: int) -> int:
        counts = [0] * 34
        for tile in hand:
            k = _CONVERTER[tile]
            counts[k] += 1
        return self.__impl.calculate(counts, n)
