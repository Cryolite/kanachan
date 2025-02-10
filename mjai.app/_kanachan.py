import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tensordict import TensorDict  # type: ignore
import torch
from hand_calculator import HandCalculator

import nyanten
from kanachan.constants import (
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_SPARSE_FEATURES,
)
from kanachan.model_loader import load_model

_NUM2TILE = (
    "5mr",  # 0
    "1m",  # 1
    "2m",  # 2
    "3m",  # 3
    "4m",  # 4
    "5m",  # 5
    "6m",  # 6
    "7m",  # 7
    "8m",  # 8
    "9m",  # 9
    "5pr",  # 10
    "1p",  # 11
    "2p",  # 12
    "3p",  # 13
    "4p",  # 14
    "5p",  # 15
    "6p",  # 16
    "7p",  # 17
    "8p",  # 18
    "9p",  # 19
    "5sr",  # 20
    "1s",  # 21
    "2s",  # 22
    "3s",  # 23
    "4s",  # 24
    "5s",  # 25
    "6s",  # 26
    "7s",  # 27
    "8s",  # 28
    "9s",  # 29
    "E",  # 30
    "S",  # 31
    "W",  # 32
    "N",  # 33
    "P",  # 34
    "F",  # 35
    "C",  # 36
)

_TILE2NUM: dict[str, int] = {
    "5mr": 0,
    "1m": 1,
    "2m": 2,
    "3m": 3,
    "4m": 4,
    "5m": 5,
    "6m": 6,
    "7m": 7,
    "8m": 8,
    "9m": 9,
    "5pr": 10,
    "1p": 11,
    "2p": 12,
    "3p": 13,
    "4p": 14,
    "5p": 15,
    "6p": 16,
    "7p": 17,
    "8p": 18,
    "9p": 19,
    "5sr": 20,
    "1s": 21,
    "2s": 22,
    "3s": 23,
    "4s": 24,
    "5s": 25,
    "6s": 26,
    "7s": 27,
    "8s": 28,
    "9s": 29,
    "E": 30,
    "S": 31,
    "W": 32,
    "N": 33,
    "P": 34,
    "F": 35,
    "C": 36,
}

_TILE_OFFSETS = (
    0,
    1,
    5,
    9,
    13,
    17,
    20,
    24,
    28,
    32,
    36,
    37,
    41,
    45,
    49,
    53,
    56,
    60,
    64,
    68,
    72,
    73,
    77,
    81,
    85,
    89,
    92,
    96,
    100,
    104,
    108,
    112,
    116,
    120,
    124,
    128,
    132,
    136,
)

_NUM2CHI = (
    ("1m", ["2m", "3m"]),  # 0
    ("2m", ["1m", "3m"]),  # 1
    ("2m", ["3m", "4m"]),  # 2
    ("3m", ["1m", "2m"]),  # 3
    ("3m", ["2m", "4m"]),  # 4
    ("3m", ["4m", "5m"]),  # 5
    ("3m", ["4m", "5mr"]),  # 6
    ("4m", ["2m", "3m"]),  # 7
    ("4m", ["3m", "5m"]),  # 8
    ("4m", ["3m", "5mr"]),  # 9
    ("4m", ["5m", "6m"]),  # 10
    ("4m", ["5mr", "6m"]),  # 11
    ("5m", ["3m", "4m"]),  # 12
    ("5mr", ["3m", "4m"]),  # 13
    ("5m", ["4m", "6m"]),  # 14
    ("5mr", ["4m", "6m"]),  # 15
    ("5m", ["6m", "7m"]),  # 16
    ("5mr", ["6m", "7m"]),  # 17
    ("6m", ["4m", "5m"]),  # 18
    ("6m", ["4m", "5mr"]),  # 19
    ("6m", ["5m", "7m"]),  # 20
    ("6m", ["5mr", "7m"]),  # 21
    ("6m", ["7m", "8m"]),  # 22
    ("7m", ["5m", "6m"]),  # 23
    ("7m", ["5mr", "6m"]),  # 24
    ("7m", ["6m", "8m"]),  # 25
    ("7m", ["8m", "9m"]),  # 26
    ("8m", ["6m", "7m"]),  # 27
    ("8m", ["7m", "9m"]),  # 28
    ("9m", ["7m", "8m"]),  # 29
    ("1p", ["2p", "3p"]),  # 30
    ("2p", ["1p", "3p"]),  # 31
    ("2p", ["3p", "4p"]),  # 32
    ("3p", ["1p", "2p"]),  # 33
    ("3p", ["2p", "4p"]),  # 34
    ("3p", ["4p", "5p"]),  # 35
    ("3p", ["4p", "5pr"]),  # 36
    ("4p", ["2p", "3p"]),  # 37
    ("4p", ["3p", "5p"]),  # 38
    ("4p", ["3p", "5pr"]),  # 39
    ("4p", ["5p", "6p"]),  # 40
    ("4p", ["5pr", "6p"]),  # 41
    ("5p", ["3p", "4p"]),  # 42
    ("5pr", ["3p", "4p"]),  # 43
    ("5p", ["4p", "6p"]),  # 44
    ("5pr", ["4p", "6p"]),  # 45
    ("5p", ["6p", "7p"]),  # 46
    ("5pr", ["6p", "7p"]),  # 47
    ("6p", ["4p", "5p"]),  # 48
    ("6p", ["4p", "5pr"]),  # 49
    ("6p", ["5p", "7p"]),  # 50
    ("6p", ["5pr", "7p"]),  # 51
    ("6p", ["7p", "8p"]),  # 52
    ("7p", ["5p", "6p"]),  # 53
    ("7p", ["5pr", "6p"]),  # 54
    ("7p", ["6p", "8p"]),  # 55
    ("7p", ["8p", "9p"]),  # 56
    ("8p", ["6p", "7p"]),  # 57
    ("8p", ["7p", "9p"]),  # 58
    ("9p", ["7p", "8p"]),  # 59
    ("1s", ["2s", "3s"]),  # 60
    ("2s", ["1s", "3s"]),  # 61
    ("2s", ["3s", "4s"]),  # 62
    ("3s", ["1s", "2s"]),  # 63
    ("3s", ["2s", "4s"]),  # 64
    ("3s", ["4s", "5s"]),  # 65
    ("3s", ["4s", "5sr"]),  # 66
    ("4s", ["2s", "3s"]),  # 67
    ("4s", ["3s", "5s"]),  # 68
    ("4s", ["3s", "5sr"]),  # 69
    ("4s", ["5s", "6s"]),  # 70
    ("4s", ["5sr", "6s"]),  # 71
    ("5s", ["3s", "4s"]),  # 72
    ("5sr", ["3s", "4s"]),  # 73
    ("5s", ["4s", "6s"]),  # 74
    ("5sr", ["4s", "6s"]),  # 75
    ("5s", ["6s", "7s"]),  # 76
    ("5sr", ["6s", "7s"]),  # 77
    ("6s", ["4s", "5s"]),  # 78
    ("6s", ["4s", "5sr"]),  # 79
    ("6s", ["5s", "7s"]),  # 80
    ("6s", ["5sr", "7s"]),  # 81
    ("6s", ["7s", "8s"]),  # 82
    ("7s", ["5s", "6s"]),  # 83
    ("7s", ["5sr", "6s"]),  # 84
    ("7s", ["6s", "8s"]),  # 85
    ("7s", ["8s", "9s"]),  # 86
    ("8s", ["6s", "7s"]),  # 87
    ("8s", ["7s", "9s"]),  # 88
    ("9s", ["7s", "8s"]),  # 89
)

_CHI2NUM: dict[tuple[str, tuple[str, str]], int] = {
    ("1m", ("2m", "3m")): 0,
    ("2m", ("1m", "3m")): 1,
    ("2m", ("3m", "4m")): 2,
    ("3m", ("1m", "2m")): 3,
    ("3m", ("2m", "4m")): 4,
    ("3m", ("4m", "5m")): 5,
    ("3m", ("4m", "5mr")): 6,
    ("4m", ("2m", "3m")): 7,
    ("4m", ("3m", "5m")): 8,
    ("4m", ("3m", "5mr")): 9,
    ("4m", ("5m", "6m")): 10,
    ("4m", ("5mr", "6m")): 11,
    ("5m", ("3m", "4m")): 12,
    ("5mr", ("3m", "4m")): 13,
    ("5m", ("4m", "6m")): 14,
    ("5mr", ("4m", "6m")): 15,
    ("5m", ("6m", "7m")): 16,
    ("5mr", ("6m", "7m")): 17,
    ("6m", ("4m", "5m")): 18,
    ("6m", ("4m", "5mr")): 19,
    ("6m", ("5m", "7m")): 20,
    ("6m", ("5mr", "7m")): 21,
    ("6m", ("7m", "8m")): 22,
    ("7m", ("5m", "6m")): 23,
    ("7m", ("5mr", "6m")): 24,
    ("7m", ("6m", "8m")): 25,
    ("7m", ("8m", "9m")): 26,
    ("8m", ("6m", "7m")): 27,
    ("8m", ("7m", "9m")): 28,
    ("9m", ("7m", "8m")): 29,
    ("1p", ("2p", "3p")): 30,
    ("2p", ("1p", "3p")): 31,
    ("2p", ("3p", "4p")): 32,
    ("3p", ("1p", "2p")): 33,
    ("3p", ("2p", "4p")): 34,
    ("3p", ("4p", "5p")): 35,
    ("3p", ("4p", "5pr")): 36,
    ("4p", ("2p", "3p")): 37,
    ("4p", ("3p", "5p")): 38,
    ("4p", ("3p", "5pr")): 39,
    ("4p", ("5p", "6p")): 40,
    ("4p", ("5pr", "6p")): 41,
    ("5p", ("3p", "4p")): 42,
    ("5pr", ("3p", "4p")): 43,
    ("5p", ("4p", "6p")): 44,
    ("5pr", ("4p", "6p")): 45,
    ("5p", ("6p", "7p")): 46,
    ("5pr", ("6p", "7p")): 47,
    ("6p", ("4p", "5p")): 48,
    ("6p", ("4p", "5pr")): 49,
    ("6p", ("5p", "7p")): 50,
    ("6p", ("5pr", "7p")): 51,
    ("6p", ("7p", "8p")): 52,
    ("7p", ("5p", "6p")): 53,
    ("7p", ("5pr", "6p")): 54,
    ("7p", ("6p", "8p")): 55,
    ("7p", ("8p", "9p")): 56,
    ("8p", ("6p", "7p")): 57,
    ("8p", ("7p", "9p")): 58,
    ("9p", ("7p", "8p")): 59,
    ("1s", ("2s", "3s")): 60,
    ("2s", ("1s", "3s")): 61,
    ("2s", ("3s", "4s")): 62,
    ("3s", ("1s", "2s")): 63,
    ("3s", ("2s", "4s")): 64,
    ("3s", ("4s", "5s")): 65,
    ("3s", ("4s", "5sr")): 66,
    ("4s", ("2s", "3s")): 67,
    ("4s", ("3s", "5s")): 68,
    ("4s", ("3s", "5sr")): 69,
    ("4s", ("5s", "6s")): 70,
    ("4s", ("5sr", "6s")): 71,
    ("5s", ("3s", "4s")): 72,
    ("5sr", ("3s", "4s")): 73,
    ("5s", ("4s", "6s")): 74,
    ("5sr", ("4s", "6s")): 75,
    ("5s", ("6s", "7s")): 76,
    ("5sr", ("6s", "7s")): 77,
    ("6s", ("4s", "5s")): 78,
    ("6s", ("4s", "5sr")): 79,
    ("6s", ("5s", "7s")): 80,
    ("6s", ("5sr", "7s")): 81,
    ("6s", ("7s", "8s")): 82,
    ("7s", ("5s", "6s")): 83,
    ("7s", ("5sr", "6s")): 84,
    ("7s", ("6s", "8s")): 85,
    ("7s", ("8s", "9s")): 86,
    ("8s", ("6s", "7s")): 87,
    ("8s", ("7s", "9s")): 88,
    ("9s", ("7s", "8s")): 89,
}

_CHI_COUNTS = (
    (1, {2: 1, 3: 1}),  # 0
    (2, {1: 1, 3: 1}),  # 1
    (2, {3: 1, 4: 1}),  # 2
    (3, {1: 1, 2: 1}),  # 3
    (3, {2: 1, 4: 1}),  # 4
    (3, {4: 1, 5: 1}),  # 5
    (3, {4: 1, 0: 1}),  # 6
    (4, {2: 1, 3: 1}),  # 7
    (4, {3: 1, 5: 1}),  # 8
    (4, {3: 1, 0: 1}),  # 9
    (4, {5: 1, 6: 1}),  # 10
    (4, {0: 1, 6: 1}),  # 11
    (5, {3: 1, 4: 1}),  # 12
    (0, {3: 1, 4: 1}),  # 13
    (5, {4: 1, 6: 1}),  # 14
    (0, {4: 1, 6: 1}),  # 15
    (5, {6: 1, 7: 1}),  # 16
    (0, {6: 1, 7: 1}),  # 17
    (6, {4: 1, 5: 1}),  # 18
    (6, {4: 1, 0: 1}),  # 19
    (6, {5: 1, 7: 1}),  # 20
    (6, {0: 1, 7: 1}),  # 21
    (6, {7: 1, 8: 1}),  # 22
    (7, {5: 1, 6: 1}),  # 23
    (7, {0: 1, 6: 1}),  # 24
    (7, {6: 1, 8: 1}),  # 25
    (7, {8: 1, 9: 1}),  # 26
    (8, {6: 1, 7: 1}),  # 27
    (8, {7: 1, 9: 1}),  # 28
    (9, {7: 1, 8: 1}),  # 29
    (11, {12: 1, 13: 1}),  # 30
    (12, {11: 1, 13: 1}),  # 31
    (12, {13: 1, 14: 1}),  # 32
    (13, {11: 1, 12: 1}),  # 33
    (13, {12: 1, 14: 1}),  # 34
    (13, {14: 1, 15: 1}),  # 35
    (13, {14: 1, 10: 1}),  # 36
    (14, {12: 1, 13: 1}),  # 37
    (14, {13: 1, 15: 1}),  # 38
    (14, {13: 1, 10: 1}),  # 39
    (14, {15: 1, 16: 1}),  # 40
    (14, {10: 1, 16: 1}),  # 41
    (15, {13: 1, 14: 1}),  # 42
    (10, {13: 1, 14: 1}),  # 43
    (15, {14: 1, 16: 1}),  # 44
    (10, {14: 1, 16: 1}),  # 45
    (15, {16: 1, 17: 1}),  # 46
    (10, {16: 1, 17: 1}),  # 47
    (16, {14: 1, 15: 1}),  # 48
    (16, {14: 1, 10: 1}),  # 49
    (16, {15: 1, 17: 1}),  # 50
    (16, {10: 1, 17: 1}),  # 51
    (16, {17: 1, 18: 1}),  # 52
    (17, {15: 1, 16: 1}),  # 53
    (17, {10: 1, 16: 1}),  # 54
    (17, {16: 1, 18: 1}),  # 55
    (17, {18: 1, 19: 1}),  # 56
    (18, {16: 1, 17: 1}),  # 57
    (18, {17: 1, 19: 1}),  # 58
    (19, {17: 1, 18: 1}),  # 59
    (21, {22: 1, 23: 1}),  # 60
    (22, {21: 1, 23: 1}),  # 61
    (22, {23: 1, 24: 1}),  # 62
    (23, {21: 1, 22: 1}),  # 63
    (23, {22: 1, 24: 1}),  # 64
    (23, {24: 1, 25: 1}),  # 65
    (23, {24: 1, 20: 1}),  # 66
    (24, {22: 1, 23: 1}),  # 67
    (24, {23: 1, 25: 1}),  # 68
    (24, {23: 1, 20: 1}),  # 69
    (24, {25: 1, 26: 1}),  # 70
    (24, {20: 1, 26: 1}),  # 71
    (25, {23: 1, 24: 1}),  # 72
    (20, {23: 1, 24: 1}),  # 73
    (25, {24: 1, 26: 1}),  # 74
    (20, {24: 1, 26: 1}),  # 75
    (25, {26: 1, 27: 1}),  # 76
    (20, {26: 1, 27: 1}),  # 77
    (26, {24: 1, 25: 1}),  # 78
    (26, {24: 1, 20: 1}),  # 79
    (26, {25: 1, 27: 1}),  # 80
    (26, {20: 1, 27: 1}),  # 81
    (26, {27: 1, 28: 1}),  # 82
    (27, {25: 1, 26: 1}),  # 83
    (27, {20: 1, 26: 1}),  # 84
    (27, {26: 1, 28: 1}),  # 85
    (27, {28: 1, 29: 1}),  # 86
    (28, {26: 1, 27: 1}),  # 87
    (28, {27: 1, 29: 1}),  # 88
    (29, {27: 1, 28: 1}),  # 89
)

_CHI_TO_KUIKAE_TILES = (
    (1, 4),  # (2m, 3m, 1m) => 1m, 4m
    (2,),  # (1m, 3m, 2m) => 2m
    (2, 0, 5),  # (3m, 4m, 2m) => 2m, 0m, 5m
    (3,),  # (1m, 2m, 3m) => 3m
    (3,),  # (2m, 4m, 3m) => 3m
    (3, 6),  # (4m, 5m, 3m) => 3m, 6m
    (3, 6),  # (4m, 0m, 3m) => 3m, 6m
    (1, 4),  # (2m, 3m, 4m) => 1m, 4m
    (4,),  # (3m, 5m, 4m) => 4m
    (4,),  # (3m, 0m, 4m) => 4m
    (4, 7),  # (5m, 6m, 4m) => 4m, 7m
    (4, 7),  # (0m, 6m, 4m) => 4m, 7m
    (0, 2, 5),  # (3m, 4m, 5m) => 2m, 0m, 5m
    (2, 5),  # (3m, 4m, 0m) => 2m, 5m
    (0, 5),  # (4m, 6m, 5m) => 0m, 5m
    (5,),  # (4m, 6m, 0m) => 5m
    (0, 5, 8),  # (6m, 7m, 5m) => 0m, 5m, 8m
    (5, 8),  # (6m, 7m, 0m) => 5m, 8m
    (3, 6),  # (4m, 5m, 6m) => 3m, 6m
    (3, 6),  # (4m, 0m, 6m) => 3m, 6m
    (6,),  # (5m, 7m, 6m) => 6m
    (6,),  # (0m, 7m, 6m) => 6m
    (6, 9),  # (7m, 8m, 6m) => 6m, 9m
    (4, 7),  # (5m, 6m, 7m) => 4m, 7m
    (4, 7),  # (0m, 6m, 7m) => 4m, 7m
    (7,),  # (6m, 8m, 7m) => 7m
    (7,),  # (8m, 9m, 7m) => 7m
    (0, 5, 8),  # (6m, 7m, 8m) => 0m, 5m, 8m
    (8,),  # (7m, 9m, 8m) => 8m
    (6, 9),  # (7m, 8m, 9m) => 9m
    (11, 14),  # (2p, 3p, 1p) => 1p, 4p
    (12,),  # (1p, 3p, 2p) => 2p
    (12, 10, 15),  # (3p, 4p, 2p) => 2p, 0p, 5p
    (13,),  # (1p, 2p, 3p) => 3p
    (13,),  # (2p, 4p, 3p) => 3p
    (13, 16),  # (4p, 5p, 3p) => 3p, 6p
    (13, 16),  # (4p, 0p, 3p) => 3p, 6p
    (11, 14),  # (2p, 3p, 4p) => 1p, 4p
    (14,),  # (3p, 5p, 4p) => 4p
    (14,),  # (3p, 0p, 4p) => 4p
    (14, 17),  # (5p, 6p, 4p) => 4p, 7p
    (14, 17),  # (0p, 6p, 4p) => 4p, 7p
    (10, 12, 15),  # (3p, 4p, 5p) => 2p, 0p, 5p
    (12, 15),  # (3p, 4p, 0p) => 2p, 5p
    (10, 15),  # (4p, 6p, 5p) => 0p, 5p
    (15,),  # (4p, 6p, 0p) => 5p
    (10, 15, 18),  # (6p, 7p, 5p) => 0p, 5p, 8p
    (15, 18),  # (6p, 7p, 0p) => 5p, 8p
    (13, 16),  # (4p, 5p, 6p) => 3p, 6p
    (13, 16),  # (4p, 0p, 6p) => 3p, 6p
    (16,),  # (5p, 7p, 6p) => 6p
    (16,),  # (0p, 7p, 6p) => 6p
    (16, 19),  # (7p, 8p, 6p) => 6p, 9p
    (14, 17),  # (5p, 6p, 7p) => 4p, 7p
    (14, 17),  # (0p, 6p, 7p) => 4p, 7p
    (17,),  # (6p, 8p, 7p) => 7p
    (17,),  # (8p, 9p, 7p) => 7p
    (10, 15, 18),  # (6p, 7p, 8p) => 0p, 5p, 8p
    (18,),  # (7p, 9p, 8p) => 8p
    (16, 19),  # (7p, 8p, 9p) => 9p
    (21, 24),  # (2s, 3s, 1s) => 1s, 4s
    (22,),  # (1s, 3s, 2s) => 2s
    (22, 20, 25),  # (3s, 4s, 2s) => 2s, 0s, 5s
    (23,),  # (1s, 2s, 3s) => 3s
    (23,),  # (2s, 4s, 3s) => 3s
    (23, 26),  # (4s, 5s, 3s) => 3s, 6s
    (23, 26),  # (4s, 0s, 3s) => 3s, 6s
    (21, 24),  # (2s, 3s, 4s) => 1s, 4s
    (24,),  # (3s, 5s, 4s) => 4s
    (24,),  # (3s, 0s, 4s) => 4s
    (24, 27),  # (5s, 6s, 4s) => 4s, 7s
    (24, 27),  # (0s, 6s, 4s) => 4s, 7s
    (20, 22, 25),  # (3s, 4s, 5s) => 2s, 0s, 5s
    (22, 25),  # (3s, 4s, 0s) => 2s, 5s
    (20, 25),  # (4s, 6s, 5s) => 0s, 5s
    (25,),  # (4s, 6s, 0s) => 5s
    (20, 25, 28),  # (6s, 7s, 5s) => 0s, 5s, 8s
    (25, 28),  # (6s, 7s, 0s) => 5s, 8s
    (23, 26),  # (4s, 5s, 6s) => 3s, 6s
    (23, 26),  # (4s, 0s, 6s) => 3s, 6s
    (26,),  # (5s, 7s, 6s) => 6s
    (26,),  # (0s, 7s, 6s) => 6s
    (26, 29),  # (7s, 8s, 6s) => 6s, 9s
    (24, 27),  # (5s, 6s, 7s) => 4s, 7s
    (24, 27),  # (0s, 6s, 7s) => 4s, 7s
    (27,),  # (6s, 8s, 7s) => 7s
    (27,),  # (8s, 9s, 7s) => 7s
    (20, 25, 28),  # (6s, 7s, 8s) => 0s, 5s, 8s
    (28,),  # (7s, 9s, 8s) => 8s
    (26, 29),  # (7s, 8s, 9s) => 9s
)

_NUM2PENG = (
    ("1m", ["1m", "1m"]),  # 0
    ("2m", ["2m", "2m"]),  # 1
    ("3m", ["3m", "3m"]),  # 2
    ("4m", ["4m", "4m"]),  # 3
    ("5m", ["5m", "5m"]),  # 4
    ("5m", ["5m", "5mr"]),  # 5
    ("5mr", ["5m", "5m"]),  # 6
    ("6m", ["6m", "6m"]),  # 7
    ("7m", ["7m", "7m"]),  # 8
    ("8m", ["8m", "8m"]),  # 9
    ("9m", ["9m", "9m"]),  # 10
    ("1p", ["1p", "1p"]),  # 11
    ("2p", ["2p", "2p"]),  # 12
    ("3p", ["3p", "3p"]),  # 13
    ("4p", ["4p", "4p"]),  # 14
    ("5p", ["5p", "5p"]),  # 15
    ("5p", ["5p", "5pr"]),  # 16
    ("5pr", ["5p", "5p"]),  # 17
    ("6p", ["6p", "6p"]),  # 18
    ("7p", ["7p", "7p"]),  # 19
    ("8p", ["8p", "8p"]),  # 20
    ("9p", ["9p", "9p"]),  # 21
    ("1s", ["1s", "1s"]),  # 22
    ("2s", ["2s", "2s"]),  # 23
    ("3s", ["3s", "3s"]),  # 24
    ("4s", ["4s", "4s"]),  # 25
    ("5s", ["5s", "5s"]),  # 26
    ("5s", ["5s", "5sr"]),  # 27
    ("5sr", ["5s", "5s"]),  # 28
    ("6s", ["6s", "6s"]),  # 29
    ("7s", ["7s", "7s"]),  # 30
    ("8s", ["8s", "8s"]),  # 31
    ("9s", ["9s", "9s"]),  # 32
    ("E", ["E", "E"]),  # 33
    ("S", ["S", "S"]),  # 34
    ("W", ["W", "W"]),  # 35
    ("N", ["N", "N"]),  # 36
    ("P", ["P", "P"]),  # 37
    ("F", ["F", "F"]),  # 38
    ("C", ["C", "C"]),  # 39
)

_PENG2NUM: dict[tuple[str, tuple[str, str]], int] = {
    ("1m", ("1m", "1m")): 0,
    ("2m", ("2m", "2m")): 1,
    ("3m", ("3m", "3m")): 2,
    ("4m", ("4m", "4m")): 3,
    ("5m", ("5m", "5m")): 4,
    ("5m", ("5m", "5mr")): 5,
    ("5mr", ("5m", "5m")): 6,
    ("6m", ("6m", "6m")): 7,
    ("7m", ("7m", "7m")): 8,
    ("8m", ("8m", "8m")): 9,
    ("9m", ("9m", "9m")): 10,
    ("1p", ("1p", "1p")): 11,
    ("2p", ("2p", "2p")): 12,
    ("3p", ("3p", "3p")): 13,
    ("4p", ("4p", "4p")): 14,
    ("5p", ("5p", "5p")): 15,
    ("5p", ("5p", "5pr")): 16,
    ("5pr", ("5p", "5p")): 17,
    ("6p", ("6p", "6p")): 18,
    ("7p", ("7p", "7p")): 19,
    ("8p", ("8p", "8p")): 20,
    ("9p", ("9p", "9p")): 21,
    ("1s", ("1s", "1s")): 22,
    ("2s", ("2s", "2s")): 23,
    ("3s", ("3s", "3s")): 24,
    ("4s", ("4s", "4s")): 25,
    ("5s", ("5s", "5s")): 26,
    ("5s", ("5s", "5sr")): 27,
    ("5sr", ("5s", "5s")): 28,
    ("6s", ("6s", "6s")): 29,
    ("7s", ("7s", "7s")): 30,
    ("8s", ("8s", "8s")): 31,
    ("9s", ("9s", "9s")): 32,
    ("E", ("E", "E")): 33,
    ("S", ("S", "S")): 34,
    ("W", ("W", "W")): 35,
    ("N", ("N", "N")): 36,
    ("P", ("P", "P")): 37,
    ("F", ("F", "F")): 38,
    ("C", ("C", "C")): 39,
}

_PENG_COUNTS = (
    (1, {1: 2}),  # 0
    (2, {2: 2}),  # 1
    (3, {3: 2}),  # 2
    (4, {4: 2}),  # 3
    (5, {5: 2}),  # 4
    (5, {0: 1, 5: 1}),  # 5
    (0, {5: 2}),  # 6
    (6, {6: 2}),  # 7
    (7, {7: 2}),  # 8
    (8, {8: 2}),  # 9
    (9, {9: 2}),  # 10
    (11, {11: 2}),  # 11
    (12, {12: 2}),  # 12
    (13, {13: 2}),  # 13
    (14, {14: 2}),  # 14
    (15, {15: 2}),  # 15
    (15, {10: 1, 15: 1}),  # 16
    (10, {15: 2}),  # 17
    (16, {16: 2}),  # 18
    (17, {17: 2}),  # 19
    (18, {18: 2}),  # 20
    (19, {19: 2}),  # 21
    (21, {21: 2}),  # 22
    (22, {22: 2}),  # 23
    (23, {23: 2}),  # 24
    (24, {24: 2}),  # 25
    (25, {25: 2}),  # 26
    (25, {20: 1, 25: 1}),  # 27
    (20, {25: 2}),  # 28
    (26, {26: 2}),  # 29
    (27, {27: 2}),  # 30
    (28, {28: 2}),  # 31
    (29, {29: 2}),  # 32
    (30, {30: 2}),  # 33
    (31, {31: 2}),  # 34
    (32, {32: 2}),  # 35
    (33, {33: 2}),  # 36
    (34, {34: 2}),  # 37
    (35, {35: 2}),  # 38
    (36, {36: 2}),  # 39
)

_PENG_TO_KUIKAE_TILE = (
    1,  # (1m, 1m, 1m) => 1m
    2,  # (2m, 2m, 2m) => 2m
    3,  # (3m, 3m, 3m) => 3m
    4,  # (4m, 4m, 4m) => 4m
    0,  # (5m, 5m, 5m) => 0m
    5,  # (0m, 5m, 5m) => 5m
    5,  # (5m, 5m, 0m) => 5m
    6,  # (6m, 6m, 6m) => 6m
    7,  # (7m, 7m, 7m) => 7m
    8,  # (8m, 8m, 8m) => 8m
    9,  # (9m, 9m, 9m) => 9m
    11,  # (1p, 1p, 1p) => 1p
    12,  # (2p, 2p, 2p) => 2p
    13,  # (3p, 3p, 3p) => 3p
    14,  # (4p, 4p, 4p) => 4p
    10,  # (5p, 5p, 5p) => 0p
    15,  # (0p, 5p, 5p) => 5p
    15,  # (5p, 5p, 0p) => 5p
    16,  # (6p, 6p, 6p) => 6p
    17,  # (7p, 7p, 7p) => 7p
    18,  # (8p, 8p, 8p) => 8p
    19,  # (9p, 9p, 9p) => 9p
    21,  # (1s, 1s, 1s) => 1s
    22,  # (2s, 2s, 2s) => 2s
    23,  # (3s, 3s, 3s) => 3s
    24,  # (4s, 4s, 4s) => 4s
    20,  # (5s, 5s, 5s) => 0s
    25,  # (0s, 5s, 5s) => 5s
    25,  # (5s, 5s, 0s) => 5s
    26,  # (6s, 6s, 6s) => 6s
    27,  # (7s, 7s, 7s) => 7s
    28,  # (8s, 8s, 8s) => 8s
    29,  # (9s, 9s, 9s) => 9s
    30,  # (1z, 1z, 1z) => 1z
    31,  # (2z, 2z, 2z) => 2z
    32,  # (3z, 3z, 3z) => 3z
    33,  # (4z, 4z, 4z) => 4z
    34,  # (5z, 5z, 5z) => 5z
    35,  # (6z, 6z, 6z) => 6z
    36,  # (7z, 7z, 7z) => 7z
)

_NUM2DAMINGGANG = (
    ("5mr", ["5m", "5m", "5m"]),  # 0
    ("1m", ["1m", "1m", "1m"]),  # 1
    ("2m", ["2m", "2m", "2m"]),  # 2
    ("3m", ["3m", "3m", "3m"]),  # 3
    ("4m", ["4m", "4m", "4m"]),  # 4
    ("5m", ["5m", "5m", "5mr"]),  # 5
    ("6m", ["6m", "6m", "6m"]),  # 6
    ("7m", ["7m", "7m", "7m"]),  # 7
    ("8m", ["8m", "8m", "8m"]),  # 8
    ("9m", ["9m", "9m", "9m"]),  # 9
    ("5pr", ["5p", "5p", "5p"]),  # 10
    ("1p", ["1p", "1p", "1p"]),  # 11
    ("2p", ["2p", "2p", "2p"]),  # 12
    ("3p", ["3p", "3p", "3p"]),  # 13
    ("4p", ["4p", "4p", "4p"]),  # 14
    ("5p", ["5p", "5p", "5pr"]),  # 15
    ("6p", ["6p", "6p", "6p"]),  # 16
    ("7p", ["7p", "7p", "7p"]),  # 17
    ("8p", ["8p", "8p", "8p"]),  # 18
    ("9p", ["9p", "9p", "9p"]),  # 19
    ("5sr", ["5s", "5s", "5s"]),  # 20
    ("1s", ["1s", "1s", "1s"]),  # 21
    ("2s", ["2s", "2s", "2s"]),  # 22
    ("3s", ["3s", "3s", "3s"]),  # 23
    ("4s", ["4s", "4s", "4s"]),  # 24
    ("5s", ["5s", "5s", "5sr"]),  # 25
    ("6s", ["6s", "6s", "6s"]),  # 26
    ("7s", ["7s", "7s", "7s"]),  # 27
    ("8s", ["8s", "8s", "8s"]),  # 28
    ("9s", ["9s", "9s", "9s"]),  # 29
    ("E", ["E", "E", "E"]),  # 30
    ("S", ["S", "S", "S"]),  # 31
    ("W", ["W", "W", "W"]),  # 32
    ("N", ["N", "N", "N"]),  # 33
    ("P", ["P", "P", "P"]),  # 34
    ("F", ["F", "F", "F"]),  # 35
    ("C", ["C", "C", "C"]),  # 36
)

_DAMINGGANG2NUM: dict[tuple[str, tuple[str, str, str]], int] = {
    ("5mr", ("5m", "5m", "5m")): 0,
    ("1m", ("1m", "1m", "1m")): 1,
    ("2m", ("2m", "2m", "2m")): 2,
    ("3m", ("3m", "3m", "3m")): 3,
    ("4m", ("4m", "4m", "4m")): 4,
    ("5m", ("5m", "5m", "5mr")): 5,
    ("6m", ("6m", "6m", "6m")): 6,
    ("7m", ("7m", "7m", "7m")): 7,
    ("8m", ("8m", "8m", "8m")): 8,
    ("9m", ("9m", "9m", "9m")): 9,
    ("5pr", ("5p", "5p", "5p")): 10,
    ("1p", ("1p", "1p", "1p")): 11,
    ("2p", ("2p", "2p", "2p")): 12,
    ("3p", ("3p", "3p", "3p")): 13,
    ("4p", ("4p", "4p", "4p")): 14,
    ("5p", ("5p", "5p", "5pr")): 15,
    ("6p", ("6p", "6p", "6p")): 16,
    ("7p", ("7p", "7p", "7p")): 17,
    ("8p", ("8p", "8p", "8p")): 18,
    ("9p", ("9p", "9p", "9p")): 19,
    ("5sr", ("5s", "5s", "5s")): 20,
    ("1s", ("1s", "1s", "1s")): 21,
    ("2s", ("2s", "2s", "2s")): 22,
    ("3s", ("3s", "3s", "3s")): 23,
    ("4s", ("4s", "4s", "4s")): 24,
    ("5s", ("5s", "5s", "5sr")): 25,
    ("6s", ("6s", "6s", "6s")): 26,
    ("7s", ("7s", "7s", "7s")): 27,
    ("8s", ("8s", "8s", "8s")): 28,
    ("9s", ("9s", "9s", "9s")): 29,
    ("E", ("E", "E", "E")): 30,
    ("S", ("S", "S", "S")): 31,
    ("W", ("W", "W", "W")): 32,
    ("N", ("N", "N", "N")): 33,
    ("P", ("P", "P", "P")): 34,
    ("F", ("F", "F", "F")): 35,
    ("C", ("C", "C", "C")): 36,
}

_DAMINGGANG_COUNTS = (
    {5: 3},  # 0
    {1: 3},  # 1
    {2: 3},  # 2
    {3: 3},  # 3
    {4: 3},  # 4
    {0: 1, 5: 2},  # 5
    {6: 3},  # 6
    {7: 3},  # 7
    {8: 3},  # 8
    {9: 3},  # 9
    {15: 3},  # 10
    {11: 3},  # 11
    {12: 3},  # 12
    {13: 3},  # 13
    {14: 3},  # 14
    {10: 1, 15: 2},  # 15
    {16: 3},  # 16
    {17: 3},  # 17
    {18: 3},  # 18
    {19: 3},  # 19
    {25: 3},  # 20
    {21: 3},  # 21
    {22: 3},  # 22
    {23: 3},  # 23
    {24: 3},  # 24
    {20: 1, 25: 2},  # 25
    {26: 3},  # 26
    {27: 3},  # 27
    {28: 3},  # 28
    {29: 3},  # 29
    {30: 3},  # 30
    {31: 3},  # 31
    {32: 3},  # 32
    {33: 3},  # 33
    {34: 3},  # 34
    {35: 3},  # 35
    {36: 3},  # 36
)

_NUM2ANGANG = (
    ["1m", "1m", "1m", "1m"],  # 0
    ["2m", "2m", "2m", "2m"],  # 1
    ["3m", "3m", "3m", "3m"],  # 2
    ["4m", "4m", "4m", "4m"],  # 3
    ["5m", "5m", "5m", "5mr"],  # 4
    ["6m", "6m", "6m", "6m"],  # 5
    ["7m", "7m", "7m", "7m"],  # 6
    ["8m", "8m", "8m", "8m"],  # 7
    ["9m", "9m", "9m", "9m"],  # 8
    ["1p", "1p", "1p", "1p"],  # 9
    ["2p", "2p", "2p", "2p"],  # 10
    ["3p", "3p", "3p", "3p"],  # 11
    ["4p", "4p", "4p", "4p"],  # 12
    ["5p", "5p", "5p", "5pr"],  # 13
    ["6p", "6p", "6p", "6p"],  # 14
    ["7p", "7p", "7p", "7p"],  # 15
    ["8p", "8p", "8p", "8p"],  # 16
    ["9p", "9p", "9p", "9p"],  # 17
    ["1s", "1s", "1s", "1s"],  # 18
    ["2s", "2s", "2s", "2s"],  # 19
    ["3s", "3s", "3s", "3s"],  # 20
    ["4s", "4s", "4s", "4s"],  # 21
    ["5s", "5s", "5s", "5sr"],  # 22
    ["6s", "6s", "6s", "6s"],  # 23
    ["7s", "7s", "7s", "7s"],  # 24
    ["8s", "8s", "8s", "8s"],  # 25
    ["9s", "9s", "9s", "9s"],  # 26
    ["E", "E", "E", "E"],  # 27
    ["S", "S", "S", "S"],  # 28
    ["W", "W", "W", "W"],  # 29
    ["N", "N", "N", "N"],  # 30
    ["P", "P", "P", "P"],  # 31
    ["F", "F", "F", "F"],  # 32
    ["C", "C", "C", "C"],  # 33
)

_ANGANG2NUM: dict[tuple[str, str, str, str], int] = {
    ("1m", "1m", "1m", "1m"): 0,
    ("2m", "2m", "2m", "2m"): 1,
    ("3m", "3m", "3m", "3m"): 2,
    ("4m", "4m", "4m", "4m"): 3,
    ("5m", "5m", "5m", "5mr"): 4,
    ("6m", "6m", "6m", "6m"): 5,
    ("7m", "7m", "7m", "7m"): 6,
    ("8m", "8m", "8m", "8m"): 7,
    ("9m", "9m", "9m", "9m"): 8,
    ("1p", "1p", "1p", "1p"): 9,
    ("2p", "2p", "2p", "2p"): 10,
    ("3p", "3p", "3p", "3p"): 11,
    ("4p", "4p", "4p", "4p"): 12,
    ("5p", "5p", "5p", "5pr"): 13,
    ("6p", "6p", "6p", "6p"): 14,
    ("7p", "7p", "7p", "7p"): 15,
    ("8p", "8p", "8p", "8p"): 16,
    ("9p", "9p", "9p", "9p"): 17,
    ("1s", "1s", "1s", "1s"): 18,
    ("2s", "2s", "2s", "2s"): 19,
    ("3s", "3s", "3s", "3s"): 20,
    ("4s", "4s", "4s", "4s"): 21,
    ("5s", "5s", "5s", "5sr"): 22,
    ("6s", "6s", "6s", "6s"): 23,
    ("7s", "7s", "7s", "7s"): 24,
    ("8s", "8s", "8s", "8s"): 25,
    ("9s", "9s", "9s", "9s"): 26,
    ("E", "E", "E", "E"): 27,
    ("S", "S", "S", "S"): 28,
    ("W", "W", "W", "W"): 29,
    ("N", "N", "N", "N"): 30,
    ("P", "P", "P", "P"): 31,
    ("F", "F", "F", "F"): 32,
    ("C", "C", "C", "C"): 33,
}

_ANGANG_COUNTS = (
    {1: 4},  # 0
    {2: 4},  # 1
    {3: 4},  # 2
    {4: 4},  # 3
    {0: 1, 5: 3},  # 4
    {6: 4},  # 5
    {7: 4},  # 6
    {8: 4},  # 7
    {9: 4},  # 8
    {11: 4},  # 9
    {12: 4},  # 10
    {13: 4},  # 11
    {14: 4},  # 12
    {10: 1, 15: 3},  # 13
    {16: 4},  # 14
    {17: 4},  # 15
    {18: 4},  # 16
    {19: 4},  # 17
    {21: 4},  # 18
    {22: 4},  # 19
    {23: 4},  # 20
    {24: 4},  # 21
    {20: 1, 25: 3},  # 22
    {26: 4},  # 23
    {27: 4},  # 24
    {28: 4},  # 25
    {29: 4},  # 26
    {30: 4},  # 27
    {31: 4},  # 28
    {32: 4},  # 29
    {33: 4},  # 30
    {34: 4},  # 31
    {35: 4},  # 32
    {36: 4},  # 33
)

_NUM2JIAGANG = (
    ("5mr", ["5m", "5m", "5m"]),  # 0
    ("1m", ["1m", "1m", "1m"]),  # 1
    ("2m", ["2m", "2m", "2m"]),  # 2
    ("3m", ["3m", "3m", "3m"]),  # 3
    ("4m", ["4m", "4m", "4m"]),  # 4
    ("5m", ["5m", "5m", "5mr"]),  # 5
    ("6m", ["6m", "6m", "6m"]),  # 6
    ("7m", ["7m", "7m", "7m"]),  # 7
    ("8m", ["8m", "8m", "8m"]),  # 8
    ("9m", ["9m", "9m", "9m"]),  # 9
    ("5pr", ["5p", "5p", "5p"]),  # 10
    ("1p", ["1p", "1p", "1p"]),  # 11
    ("2p", ["2p", "2p", "2p"]),  # 12
    ("3p", ["3p", "3p", "3p"]),  # 13
    ("4p", ["4p", "4p", "4p"]),  # 14
    ("5p", ["5p", "5p", "5pr"]),  # 15
    ("6p", ["6p", "6p", "6p"]),  # 16
    ("7p", ["7p", "7p", "7p"]),  # 17
    ("8p", ["8p", "8p", "8p"]),  # 18
    ("9p", ["9p", "9p", "9p"]),  # 19
    ("5sr", ["5s", "5s", "5s"]),  # 20
    ("1s", ["1s", "1s", "1s"]),  # 21
    ("2s", ["2s", "2s", "2s"]),  # 22
    ("3s", ["3s", "3s", "3s"]),  # 23
    ("4s", ["4s", "4s", "4s"]),  # 24
    ("5s", ["5s", "5s", "5sr"]),  # 25
    ("6s", ["6s", "6s", "6s"]),  # 26
    ("7s", ["7s", "7s", "7s"]),  # 27
    ("8s", ["8s", "8s", "8s"]),  # 28
    ("9s", ["9s", "9s", "9s"]),  # 29
    ("E", ["E", "E", "E"]),  # 30
    ("S", ["S", "S", "S"]),  # 31
    ("W", ["W", "W", "W"]),  # 32
    ("N", ["N", "N", "N"]),  # 33
    ("P", ["P", "P", "P"]),  # 34
    ("F", ["F", "F", "F"]),  # 35
    ("C", ["C", "C", "C"]),
)

_PENG_TO_JIAGANG_LIST = (
    1,  #  0: (1m, 1m, 1m) + 1m
    2,  #  1: (2m, 2m, 2m) + 2m
    3,  #  2: (3m, 3m, 3m) + 3m
    4,  #  3: (4m, 4m, 4m) + 4m
    0,  #  4: (5m, 5m, 5m) + 0m
    5,  #  5: (0m, 5m, 5m) + 5m
    5,  #  6: (5m, 5m, 0m) + 5m
    6,  #  7: (6m, 6m, 6m) + 6m
    7,  #  8: (7m, 7m, 7m) + 7m
    8,  #  9: (8m, 8m, 8m) + 8m
    9,  # 10: (9m, 9m, 9m) + 9m
    11,  # 11: (1p, 1p, 1p) + 1p
    12,  # 12: (2p, 2p, 2p) + 2p
    13,  # 13: (3p, 3p, 3p) + 3p
    14,  # 14: (4p, 4p, 4p) + 4p
    10,  # 15: (5p, 5p, 5p) + 0p
    15,  # 16: (0p, 5p, 5p) + 5p
    15,  # 17: (5p, 5p, 0p) + 5p
    16,  # 18: (6p, 6p, 6p) + 6p
    17,  # 19: (7p, 7p, 7p) + 7p
    18,  # 20: (8p, 8p, 8p) + 8p
    19,  # 21: (9p, 9p, 9p) + 9p
    21,  # 22: (1s, 1s, 1s) + 1s
    22,  # 23: (2s, 2s, 2s) + 2s
    23,  # 24: (3s, 3s, 3s) + 3s
    24,  # 25: (4s, 4s, 4s) + 4s
    20,  # 26: (5s, 5s, 5s) + 0s
    25,  # 27: (0s, 5s, 5s) + 5s
    25,  # 28: (5s, 5s, 0s) + 5s
    26,  # 29: (6s, 6s, 6s) + 6s
    27,  # 30: (7s, 7s, 7s) + 7s
    28,  # 31: (8s, 8s, 8s) + 8s
    29,  # 32: (9s, 9s, 9s) + 9s
    30,  # 33: (1z, 1z, 1z) + 1z
    31,  # 34: (2z, 2z, 2z) + 2z
    32,  # 35: (3z, 3z, 3z) + 3z
    33,  # 36: (4z, 4z, 4z) + 4z
    34,  # 37: (5z, 5z, 5z) + 5z
    35,  # 38: (6z, 6z, 6z) + 6z
    36,  # 39: (7z, 7z, 7z) + 7z
)

_JIAGANG_TO_PENG_LIST = (
    (4,),
    (0,),
    (1,),
    (2,),
    (3,),
    (5, 6),
    (7,),
    (8,),
    (9,),
    (10,),
    (15,),
    (11,),
    (12,),
    (13,),
    (14,),
    (16, 17),
    (18,),
    (19,),
    (20,),
    (21,),
    (26,),
    (22,),
    (23,),
    (24,),
    (25,),
    (27, 28),
    (29,),
    (30,),
    (31,),
    (32,),
    (33,),
    (34,),
    (35,),
    (36,),
    (37,),
    (38,),
    (39,),
)

_TILE37_TO_TILE34 = (
    4,  # 0
    0,  # 1
    1,  # 2
    2,  # 3
    3,  # 4
    4,  # 5
    5,  # 6
    6,  # 7
    7,  # 8
    8,  # 9
    13,  # 10
    9,  # 11
    10,  # 12
    11,  # 13
    12,  # 14
    13,  # 15
    14,  # 16
    15,  # 17
    16,  # 18
    17,  # 19
    22,  # 20
    18,  # 21
    19,  # 22
    20,  # 23
    21,  # 24
    22,  # 25
    23,  # 26
    24,  # 27
    25,  # 28
    26,  # 29
    27,  # 30
    28,  # 31
    29,  # 32
    30,  # 33
    31,  # 34
    32,  # 35
    33,  # 36
)


def _calculate_replacement_number(hand: list[int]) -> int:
    tile_counts: list[int] = [0 for _ in range(34)]
    for tile37 in hand:
        tile34 = _TILE37_TO_TILE34[tile37]
        tile_counts[tile34] += 1
    return nyanten.calculate_replacement_number(tile_counts)  # type: ignore


def _get_hupai_candidates(hand: list[int]) -> set[int]:
    counts: Counter[int] = Counter()
    for tile37 in hand:
        tile34 = _TILE37_TO_TILE34[tile37]
        counts[tile34] += 1

    result: set[int] = set()
    for tile37 in range(37):
        tile34 = _TILE37_TO_TILE34[tile37]

        if counts[tile34] >= 4:
            # 手牌にすでに4枚ある牌は候補から除外する．
            assert counts[tile34] == 4
            continue

        new_hand = hand + [tile37]
        new_hand.sort()
        replacement_number = _calculate_replacement_number(new_hand)
        if replacement_number == 0:
            result.add(tile34)

    return result


class GameState:
    def __init__(
        self,
        *,
        my_name: str,
        room: int,
        game_style: int,
        my_grade: int,
        opponent_grade: int,
    ) -> None:
        self._my_name = my_name
        self._room = room
        self._game_style = game_style
        self._my_grade = my_grade
        self._opponent_grade = opponent_grade
        self._seat: int | None = None
        self._player_grades: list[int] | None = None
        self._player_scores: list[int] | None = None

    def on_new_game(self, seat: int) -> None:
        if seat < 0:
            raise ValueError(seat)
        if seat >= 4:
            raise ValueError(seat)
        self._seat = seat

        self._player_grades = []
        for i in range(4):
            if i == self._seat:
                self._player_grades.append(self._my_grade)
            else:
                self._player_grades.append(self._opponent_grade)

    def on_new_round(self, scores: list[int]) -> None:
        self._player_scores = list(scores)

    def _assert_initialized(self) -> None:
        if self._player_grades is None:
            raise RuntimeError(
                "A method is called on a non-initialized `GameState` object."
            )

    def on_liqi_acceptance(self, seat: int) -> None:
        self._assert_initialized()
        assert self._player_scores is not None
        self._player_scores[seat] -= 1000

    def get_my_name(self) -> str:
        self._assert_initialized()
        return self._my_name

    def get_room(self) -> int:
        self._assert_initialized()
        return self._room

    def get_game_style(self) -> int:
        self._assert_initialized()
        return self._game_style

    def get_seat(self) -> int:
        self._assert_initialized()
        assert self._seat is not None
        return self._seat

    def get_player_grade(self, seat: int) -> int:
        self._assert_initialized()
        assert self._player_grades is not None
        return self._player_grades[seat]

    def get_player_rank(self, seat: int) -> int:
        self._assert_initialized()
        assert self._player_scores is not None

        score = self._player_scores[seat]
        rank = 0
        for i in range(seat):
            if self._player_scores[i] >= score:
                rank += 1
        for i in range(seat + 1, 4):
            if self._player_scores[i] > score:
                rank += 1
        assert 0 <= rank and rank < 4
        return rank

    def get_player_score(self, seat: int) -> int:
        self._assert_initialized()
        assert self._player_scores is not None
        return self._player_scores[seat]


class RoundState:
    def __init__(self) -> None:
        self._hand_calculator = HandCalculator()
        self._chang: int = -1
        self._ju: int = -1
        self._ben_chang: int = -1
        self._deposits: int = -1
        self._dora_indicators: list[int] = []
        self._num_left_tiles: int = -1
        self._my_hand: list[int] = []
        self._my_fulu_list: list[int] = []
        self._zimo_tile: int | None = None
        self._my_first_zimo: bool = False
        self._liqi_to_be_accepted: list[bool] = []
        self._my_liqi: bool = False
        self._my_lingshang_zimo: bool = False
        self._my_kuikae_tiles: list[int] = []
        self._my_zhenting: int = -1
        self._progression: list[int] = []

    def on_new_round(
        self,
        chang: int,
        ju: int,
        ben_chang: int,
        deposits: int,
        dora_indicator: int,
        hand: list[int],
    ) -> None:
        self._chang = chang
        self._ju = ju
        self._ben_chang = ben_chang
        self._deposits = deposits
        self._dora_indicators = [dora_indicator]
        self._num_left_tiles = 70
        self._my_hand = hand
        # 148 <= self._my_fulu_list <= 181: 暗槓
        # 182 <= self._my_fulu_list <= 218: 加槓
        # 222 <= self._my_fulu_list <= 311: チー
        # 312 <= self._my_fulu_list <= 431: ポン
        # 432 <= self._my_fulu_list <= 542: 大明槓
        self._my_fulu_list = []
        self._zimo_tile = None
        # 副露が先行していない自身の第一自摸であるかどうか．
        self._my_first_zimo = True
        self._liqi_to_be_accepted = [False, False, False, False]
        self._my_liqi = False
        self._my_lingshang_zimo = False
        self._my_kuikae_tiles = []
        # self.__my_zhenting == 1: 非立直中の栄和拒否による一時的なフリテン
        # self.__my_zhenting == 2: 立直中の栄和拒否による永続的なフリテン
        self._my_zhenting = 0
        self._progression = [0]

    def get_chang(self) -> int:
        assert self._chang in (0, 1, 2)
        return self._chang

    def get_ju(self) -> int:
        assert self._ju in (0, 1, 2, 3)
        return self._ju

    def get_num_ben_chang(self) -> int:
        assert self._ben_chang != -1
        return self._ben_chang

    def get_num_deposits(self) -> int:
        assert self._deposits != -1
        return self._deposits

    def get_dora_indicators(self) -> list[int]:
        assert len(self._dora_indicators) >= 1
        assert len(self._dora_indicators) <= 5
        return self._dora_indicators

    def get_num_left_tiles(self) -> int:
        assert self._num_left_tiles >= 0
        assert self._num_left_tiles <= 70
        return self._num_left_tiles

    def get_my_hand(self) -> list[int]:
        assert len(self._my_hand) >= 1
        assert len(self._my_hand) <= 14
        assert len(self._my_hand) % 3 != 0
        return self._my_hand

    def get_my_fulu_list(self) -> list[int]:
        assert len(self._my_fulu_list) <= 4
        return self._my_fulu_list

    def get_zimo_tile(self) -> int | None:
        return self._zimo_tile

    def is_in_liqi(self) -> bool:
        return self._my_liqi

    def copy_progression(self) -> list[int]:
        assert len(self._progression) >= 1
        return list(self._progression)

    def _get_my_hand_counts(self) -> Counter[int]:
        my_hand_counts: Counter[int] = Counter()
        for tile in self._my_hand:
            my_hand_counts[tile] += 1
        return my_hand_counts

    def _set_my_hand_counts(self, hand_counts: Counter[int]) -> None:
        self._my_hand = []
        for k, v in hand_counts.items():
            if k < 0 or k >= 37:
                raise ValueError(f"An invalid tile: {k}")
            if v < 0 or v > 4:
                raise ValueError(f"An invalid count: {v}")
            for _ in range(v):
                self._my_hand.append(k)
        self._my_hand.sort()
        if len(self._my_hand) not in (1, 2, 4, 5, 7, 8, 10, 11, 13):
            raise ValueError("Invalid hand counts.")

    def _remove_tile34_from_hand(self, tile34: int) -> list[int]:
        assert tile34 >= 0
        assert tile34 < 34

        new_hand = list(self._my_hand)

        if 0 <= tile34 and tile34 <= 8:
            tile37 = tile34 + 1
            while tile37 in new_hand:
                new_hand.remove(tile37)
            if tile34 == 4 and 0 in new_hand:
                assert new_hand.count(0) == 1
                new_hand.remove(0)
            return new_hand

        if 9 <= tile34 and tile34 <= 17:
            tile37 = tile34 + 2
            while tile37 in new_hand:
                new_hand.remove(tile37)
            if tile34 == 13 and 10 in new_hand:
                assert new_hand.count(10) == 1
                new_hand.remove(10)
            return new_hand

        assert tile34 >= 18
        tile37 = tile34 + 3
        while tile37 in new_hand:
            new_hand.remove(tile37)
        if tile34 == 22 and 20 in new_hand:
            assert new_hand.count(20) == 1
            new_hand.remove(20)
        return new_hand

    def on_zimo(
        self, seat: int, mine: bool, tile: int | None, my_score: int
    ) -> list[int]:
        if self._zimo_tile is not None:
            raise ValueError(f"self.__zimo_pai = {self._zimo_tile}")
        if self._num_left_tiles <= 0:
            raise ValueError(f"self.__num_left_tiles = {self._num_left_tiles}")

        self._num_left_tiles -= 1
        self._my_kuikae_tiles = []

        if not mine:
            # This is another player's self-draw, so there's no need
            # for me to do anything.
            if tile is not None:
                raise ValueError(f"tile = {tile}")
            return []

        if tile is None:
            raise ValueError("`tile` is `None`.")
        self._zimo_tile = tile

        candidates = []

        if self._my_liqi:
            # 立直中の場合．自摸切りを候補に追加する．
            candidates.append(self._zimo_tile * 4 + 1 * 2 + 0)
        else:
            # 以下，立直中でない場合．
            for i, tile in enumerate(self._my_hand):
                # 手出しを候補として追加する．
                candidates.append(tile * 4 + 0 * 2 + 0)

                # 手出し後の手牌 `new_hand` が聴牌かどうかをチェックし，
                # 聴牌かつ面前かつ持ち点が1000点以上あれば立直が可能である．
                new_hand = list(self._my_hand)
                new_hand[i] = self._zimo_tile
                if (
                    len(self._my_fulu_list) == 0
                    and my_score >= 1000
                    and self.get_num_left_tiles() >= 4
                ):
                    replacement_number = _calculate_replacement_number(
                        new_hand
                    )
                    if replacement_number == 1:
                        # 立直宣言を伴う手出しを候補として追加する．
                        candidates.append(tile * 4 + 0 * 2 + 1)

            # 自摸切りを候補として追加する．
            candidates.append(self._zimo_tile * 4 + 1 * 2 + 0)
            if (
                len(self._my_fulu_list) == 0
                and my_score >= 1000
                and self.get_num_left_tiles() >= 4
            ):
                replacement_number = _calculate_replacement_number(
                    self._my_hand
                )
                if replacement_number == 1:
                    # 立直宣言を伴う自摸切りを候補として追加する．
                    candidates.append(self._zimo_tile * 4 + 1 * 2 + 1)

        combined_hand = self._my_hand + [self._zimo_tile]

        # 暗槓が候補として追加できるかどうかをチェックする．
        if self.get_num_left_tiles() >= 1:
            # 海底自摸でない場合のみ暗槓できる．
            counts34: Counter[int] = Counter()
            for tile37 in combined_hand:
                tile34 = _TILE37_TO_TILE34[tile37]
                counts34[tile34] += 1
            for tile34, v in counts34.items():
                if v >= 4:
                    assert v == 4

                    if self.is_in_liqi():
                        # 立直中の送り槓を禁止する．
                        if tile34 != _TILE37_TO_TILE34[self._zimo_tile]:
                            # 自摸牌以外の牌で暗槓する場合は
                            # 必ず送り槓になるため，候補に追加しない．
                            continue

                        hupai_candidates_old = _get_hupai_candidates(
                            self._my_hand
                        )
                        new_hand = self._remove_tile34_from_hand(tile34)
                        hupai_candidates_new = _get_hupai_candidates(new_hand)
                        if hupai_candidates_new != hupai_candidates_old:
                            # 待ちが変わる立直中の送り槓を禁止する．
                            continue

                    candidates.append(148 + tile34)

        # 加槓が候補として追加できるかどうかをチェックする．
        if self.get_num_left_tiles() >= 1:
            # 海底自摸意外でのみ加槓できる．
            peng_list = []
            for fulu in self._my_fulu_list:
                if 312 <= fulu and fulu <= 431:
                    peng = (fulu - 312) % 40
                    peng_list.append(peng)
            for peng, t in enumerate(_PENG_TO_JIAGANG_LIST):
                if peng in peng_list and t in combined_hand:
                    candidates.append(182 + t)

        # 自摸和が候補として追加できるかどうかをチェックする．
        replacement_number = _calculate_replacement_number(combined_hand)
        if replacement_number == 0:
            player_wind = (seat + 4 - self._ju) % 4
            has_yihan = self._hand_calculator.has_yihan(
                self._chang,
                player_wind,
                self._my_hand,
                self._my_fulu_list,
                self._zimo_tile,
                rong=False,
            )
            if (
                self._my_liqi
                or self._num_left_tiles == 0
                or self._my_lingshang_zimo
                or has_yihan
            ):
                # 立直，海底摸月，嶺上開花，
                # その他役がある場合に自摸和を候補として追加する．
                # （天和，地和は必ず面前清自摸和による1飜がある．）
                candidates.append(219)

        if self._my_first_zimo:
            assert not self._my_liqi
            # 九種九牌が候補として追加できるかどうかをチェックする．
            count = 0
            for p in set(combined_hand):
                if p in (1, 9, 11, 19, 21, 29, 30, 31, 32, 33, 34, 35, 36):
                    count += 1
            if count >= 9:
                candidates.append(220)

        self._my_first_zimo = False
        assert not any(self._liqi_to_be_accepted)
        self._my_lingshang_zimo = False
        assert len(self._my_kuikae_tiles) == 0

        candidates = list(set(candidates))
        candidates.sort()
        return candidates

    def _get_my_zhenting_tiles_34(self, seat: int) -> set[int]:
        # 自分が捨てた牌を列挙する．
        discarded_tiles_34 = set()
        for p in self._progression:
            if p < 5 or 596 < p:
                continue
            encode = p - 5
            actor = encode // 148
            encode = encode % 148
            tile37 = encode // 4
            if actor != seat:
                continue
            discarded_tiles_34.add(_TILE37_TO_TILE34[tile37])

        # 和牌の候補を列挙する．
        hupai_candidates_34 = _get_hupai_candidates(self._my_hand)

        # 和牌の候補の中に自分が捨てた牌が1つでも含まれているならば，
        # 和牌の候補全てがフリテンの対象でありロンできない．
        for hupai_candidate_34 in hupai_candidates_34:
            if hupai_candidate_34 in discarded_tiles_34:
                return hupai_candidates_34
        return set()

    def on_dapai(
        self, seat: int, actor: int, tile: int, moqi: bool
    ) -> list[int]:
        if self._num_left_tiles == 69:
            # 雀魂から学習したモデルは親の第1打牌が必ず手出しになる．
            # なお，親の第1打牌に対して鳴きが（連続して）起きた場合にも
            # コントロールフローがここに到達するが，その場合も
            # 鳴き直後の打牌であり必ず手出しであるため問題ない．
            moqi = False

        liqi = self._liqi_to_be_accepted[seat]

        encode = (
            5
            + actor * 148
            + tile * 4
            + (2 if moqi else 0)
            + (1 if liqi else 0)
        )
        self._progression.append(encode)

        if actor == seat:
            # 自分の打牌の場合．

            # 一時的なフリテンを解消する．
            if self._my_zhenting == 1:
                self._my_zhenting = 0

            if moqi:
                # 自摸切りの場合（自分が親の時における第1打牌を除く）．
                if self._zimo_tile is None:
                    raise AssertionError("TODO: (A suitable error message)")
                if self._zimo_tile != tile:
                    raise AssertionError("TODO: (A suitable error message)")
                self._zimo_tile = None
                return []
            index = None
            for i, h in enumerate(self._my_hand):
                if h == tile:
                    index = i
                    break
            if index is None:
                # 自分が親の時の第1打牌で自摸切りの場合．
                if self._num_left_tiles != 69:
                    raise RuntimeError("TODO: (A suitable error message)")
                if self._zimo_tile is None:
                    raise RuntimeError("TODO: (A suitable error message)")
                if self._zimo_tile != tile:
                    raise RuntimeError("TODO: (A suitable error message)")
                self._zimo_tile = None
                return []
            self._my_hand.pop(index)
            if self._zimo_tile is not None:
                # 自摸直後の（鳴きの直後でない）打牌の場合．
                self._my_hand.append(self._zimo_tile)
                self._zimo_tile = None
                self._my_hand.sort()
            assert len(self._my_hand) in (1, 4, 7, 10, 13)
            return []

        relseat = (actor + 4 - seat) % 4 - 1

        skippable = False

        hand_counts = self._get_my_hand_counts()

        candidates = []

        if (
            not self._my_liqi
            and relseat == 2
            and self.get_num_left_tiles() >= 1
        ):
            # チーができるかどうかチェックする．
            # 河底牌に対するチーは不可能であることに注意．
            for i, (t, consumed_counts) in enumerate(_CHI_COUNTS):
                if tile != t:
                    continue
                new_hand_counts: Counter[int] | None = Counter(hand_counts)
                for k, v in consumed_counts.items():
                    if hand_counts[k] < v:
                        new_hand_counts = None
                        break
                    assert new_hand_counts is not None
                    new_hand_counts[k] -= v
                if new_hand_counts is not None:
                    # チーの後に食い替えによって打牌が禁止される牌のみが
                    # 残る場合は，そのようなチー自体が禁止される．
                    # 以下では，そのようなチーを候補から除去している．
                    for kuikae_tile in _CHI_TO_KUIKAE_TILES[i]:
                        new_hand_counts[kuikae_tile] = 0
                    flag = False
                    for count in new_hand_counts.values():
                        if count >= 1:
                            flag = True
                            break
                    if flag:
                        self._my_kuikae_tiles = list(_CHI_TO_KUIKAE_TILES[i])
                        candidates.append(222 + i)
                        skippable = True

        if not self._my_liqi and self.get_num_left_tiles() >= 1:
            # ポンができるかどうかチェックする．
            # 河底牌に対するポンは不可能であることに注意．
            for i, (t, consumed_counts) in enumerate(_PENG_COUNTS):
                if tile != t:
                    continue
                new_hand_counts = Counter(hand_counts)
                for k, v in consumed_counts.items():
                    if hand_counts[k] < v:
                        new_hand_counts = None
                        break
                    new_hand_counts[k] -= v
                if new_hand_counts is not None:
                    # ポンの後に食い替えによって打牌が禁止される牌のみが
                    # 残る場合は，そのようなポン自体が禁止される．
                    # 以下では，そのようなポンを候補から除去している．
                    new_hand_counts[_PENG_TO_KUIKAE_TILE[i]] = 0
                    flag = False
                    for count in new_hand_counts.values():
                        if count >= 1:
                            flag = True
                            break
                    if flag:
                        self._my_kuikae_tiles = [_PENG_TO_KUIKAE_TILE[i]]
                        candidates.append(312 + relseat * 40 + i)
                        skippable = True

        if not self._my_liqi and self.get_num_left_tiles() >= 1:
            # 大明槓ができるかどうかチェックする．
            # 河底牌に対する大明槓は不可能であることに注意．．
            for t, consumed_counts in enumerate(_DAMINGGANG_COUNTS):
                if tile != t:
                    continue
                flag = True
                for k, v in consumed_counts.items():
                    if hand_counts[k] < v:
                        flag = False
                        break
                if flag:
                    candidates.append(432 + relseat * 37 + t)
                    skippable = True

        combined_hand = self._my_hand + [tile]

        replacement_number = _calculate_replacement_number(combined_hand)
        if (
            replacement_number == 0
            and _TILE37_TO_TILE34[tile]
            not in self._get_my_zhenting_tiles_34(seat)
            and self._my_zhenting == 0
        ):
            # ロンが出来るかどうかチェックする．
            player_wind = (seat + 4 - self._ju) % 4
            has_yihan = self._hand_calculator.has_yihan(
                self._chang,
                player_wind,
                self._my_hand,
                self._my_fulu_list,
                tile,
                rong=True,
            )
            if self._my_liqi or self._num_left_tiles == 0 or has_yihan:
                # 立直，海底摸月，河底撈魚，その他役がある場合に
                # ロンを候補として追加する．
                candidates.append(543 + relseat)
                skippable = True

        if replacement_number == 0 and self._my_zhenting == 0:
            # 他家の打牌を含めた結果，（役無しであっても）和了形となる場合は
            # 一時的な振聴のフラグを立てる．このフラグは，自身の次の打牌時に
            # 解消する．
            self._my_zhenting = 1

        if skippable:
            candidates.append(221)

        candidates.sort()
        return candidates

    def on_chi(self, mine: bool, seat: int, chi: int) -> list[int]:
        if chi < 0 or chi >= 90:
            raise ValueError(f"An invalid `chi` value: {chi}")

        self._my_first_zimo = False
        self._progression.append(597 + seat * 90 + chi)

        if not mine:
            self._my_kuikae_tiles = []
            return []

        my_hand_counts = self._get_my_hand_counts()
        consumed_counts = _CHI_COUNTS[chi][1]
        for k, v in consumed_counts.items():
            if my_hand_counts[k] < v:
                raise RuntimeError("An invalid chi.")
            my_hand_counts[k] -= v
        self._set_my_hand_counts(my_hand_counts)

        if len(self._my_fulu_list) == 4:
            raise RuntimeError("An invalid chi.")
        self._my_fulu_list.append(222 + chi)

        candidates = []
        for tile in self._my_hand:
            if tile not in self._my_kuikae_tiles:
                candidates.append(tile * 4 + 0 * 2 + 0)
        self._my_kuikae_tiles = []

        candidates = list(set(candidates))
        candidates.sort()
        return candidates

    def on_peng(
        self, mine: bool, seat: int, relseat: int, peng: int
    ) -> list[int]:
        if peng < 0 or peng >= 40:
            raise ValueError(f"An invalid `peng` value: {peng}")

        self._my_first_zimo = False
        self._progression.append(957 + seat * 120 + relseat * 40 + peng)

        if not mine:
            self._my_kuikae_tiles = []
            return []

        my_hand_counts = self._get_my_hand_counts()
        consumed_counts = _PENG_COUNTS[peng][1]
        for k, v in consumed_counts.items():
            if my_hand_counts[k] < v:
                raise RuntimeError("An invalid peng.")
            my_hand_counts[k] -= v
        self._set_my_hand_counts(my_hand_counts)

        if len(self._my_fulu_list) == 4:
            raise RuntimeError("An invalid peng.")
        self._my_fulu_list.append(312 + relseat * 40 + peng)

        candidates = []
        for tile in self._my_hand:
            if tile not in self._my_kuikae_tiles:
                candidates.append(tile * 4 + 0 * 2 + 0)
        self._my_kuikae_tiles = []

        candidates = list(set(candidates))
        candidates.sort()
        return candidates

    def on_daminggang(
        self, mine: bool, seat: int, relseat: int, daminggang: int
    ) -> None:
        if daminggang < 0 or daminggang >= 37:
            raise ValueError(f"An invalid `daminggang` value: {daminggang}")

        self._my_first_zimo = False
        self._my_kuikae_tiles = []
        self._progression.append(1437 + seat * 111 + relseat * 37 + daminggang)

        if not mine:
            return

        my_hand_counts = self._get_my_hand_counts()
        consumed_counts = _DAMINGGANG_COUNTS[daminggang]
        for k, v in consumed_counts.items():
            if my_hand_counts[k] < v:
                raise RuntimeError("An invalid daminggang.")
            my_hand_counts[k] -= v
        self._set_my_hand_counts(my_hand_counts)

        if len(self._my_fulu_list) == 4:
            raise RuntimeError("An invalid daminggang.")
        self._my_fulu_list.append(432 + relseat * 37 + daminggang)

        self._my_lingshang_zimo = True

    def on_angang(self, seat: int, actor: int, angang: int) -> list[int]:
        if angang < 0 or angang >= 34:
            raise ValueError(f"An invalid `angang` value: {angang}")

        self._my_first_zimo = False
        self._progression.append(1881 + actor * 34 + angang)

        if seat != actor:
            # 暗槓に対する国士無双の槍槓が可能かどうかチェックする．
            if len(self._my_fulu_list) >= 1:
                return []
            if 0 <= angang and angang <= 8:
                tile37 = angang + 1
            elif 9 <= angang and angang <= 17:
                tile37 = angang + 2
            else:
                assert 18 <= angang and angang < 34
                tile37 = angang + 3
            orphans = (1, 9, 11, 19, 21, 29, 30, 31, 32, 33, 34, 35, 36)
            if tile37 not in orphans:
                return []
            combined_hand = self._my_hand + [tile37]
            counts = 0
            for o in orphans:
                if o in combined_hand:
                    counts += 1
            if counts < 13:
                return []
            assert counts == 13
            hupai_candidates = _get_hupai_candidates(self._my_hand)
            if angang not in hupai_candidates:
                return []
            if angang in self._get_my_zhenting_tiles_34(seat):
                return []
            relseat = (actor + 4 - seat) % 4 - 1
            return [221, 543 + relseat]

        if self._zimo_tile is None:
            raise RuntimeError("TODO: (A suitable error message)")

        my_hand_counts = self._get_my_hand_counts()
        my_hand_counts[self._zimo_tile] += 1
        consumed_counts = _ANGANG_COUNTS[angang]
        for k, v in consumed_counts.items():
            if my_hand_counts[k] < v:
                raise RuntimeError("An invalid angang.")
            my_hand_counts[k] -= v
        self._set_my_hand_counts(my_hand_counts)
        self._zimo_tile = None

        if len(self._my_fulu_list) == 4:
            raise RuntimeError("An invalid angang.")
        self._my_fulu_list.append(148 + angang)

        self._my_lingshang_zimo = True

        return []

    def on_jiagang(self, seat: int, actor: int, tile: int) -> list[int]:
        if tile < 0 or tile >= 37:
            raise ValueError(f"An invalid `tile` value: {tile}")

        self._my_first_zimo = False
        self._progression.append(2017 + seat * 37 + tile)

        if seat != actor:
            # 槍槓が可能かどうかをチェックする．
            tile34 = _TILE37_TO_TILE34[tile]
            hupai_candidates = _get_hupai_candidates(self._my_hand)
            if tile34 not in hupai_candidates:
                return []
            if tile34 in self._get_my_zhenting_tiles_34(seat):
                return []
            relseat = (actor + 4 - seat) % 4 - 1
            return [221, 543 + relseat]

        if self._zimo_tile is None:
            raise RuntimeError("TODO: A suitable error message")

        index: int | None = None
        for i, h in enumerate(self._my_hand):
            if h == tile:
                index = i
                break
        if index is not None:
            # 自摸牌以外の牌を加槓した場合．
            self._my_hand.pop(index)
            self._my_hand.append(self._zimo_tile)
            self._zimo_tile = None
            self._my_hand.sort()
        else:
            # 自摸牌を加槓した場合．
            if self._zimo_tile != tile:
                raise RuntimeError("TODO: A suitable error message")
            self._zimo_tile = None

        index = None
        for i, fulu in enumerate(self._my_fulu_list):
            # 加槓の対象となるポンを探す．
            if fulu < 312 or 431 < fulu:
                # ポンではない．
                continue
            peng = (fulu - 312) % 40
            if peng in _JIAGANG_TO_PENG_LIST[tile]:
                index = i
                break
        if index is None:
            # 加槓の対象となるポンが見つからない．
            raise RuntimeError("TODO: (A suitable error message)")
        encode = self._my_fulu_list[index] - 312
        relseat = encode // 40
        peng = encode % 40
        if tile != _PENG_TO_JIAGANG_LIST[peng]:
            raise RuntimeError(tile)
        self._my_fulu_list[index] = 182 + tile

        self._my_lingshang_zimo = True

        return []

    def on_liqi(self, seat: int) -> None:
        if any(self._liqi_to_be_accepted):
            raise RuntimeError("TODO: (A suitable error message)")
        self._liqi_to_be_accepted[seat] = True

    def on_liqi_acceptance(self, mine: bool, seat: int) -> None:
        self._deposits += 1

        if not self._liqi_to_be_accepted[seat]:
            raise RuntimeError("TODO: (A suitable error message)")
        self._liqi_to_be_accepted[seat] = False

        if mine:
            self._my_liqi = True

    def on_new_dora(self, tile: int) -> None:
        if len(self._dora_indicators) >= 5:
            raise RuntimeError(self._dora_indicators)
        self._dora_indicators.append(tile)

    def set_zhenting(self, zhenting: int) -> None:
        if zhenting not in (1, 2):
            raise ValueError("TODO: (A suitable error message)")
        self._my_zhenting = zhenting


class Kanachan:
    def __init__(self) -> None:
        with open("./config.json", encoding="UTF-8") as f:
            config = json.load(f)

        model_path: Path = Path(config["model"])
        self._device = config["device"]
        self._dtype = {
            "float64": torch.float64,
            "double": torch.float64,
            "float32": torch.float32,
            "single": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
        }[config["dtype"]]
        self._model = load_model(model_path, map_location=torch.device("cpu"))
        self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()

        self._game_state = GameState(
            my_name=config["my_name"],
            room=config["room"],
            game_style=config["game_style"],
            my_grade=config["my_grade"],
            opponent_grade=config["opponent_grade"],
        )

        self._round_state = RoundState()

    def _on_hello(self, message: dict) -> None:
        assert message["type"] == "hello"

        if "can_act" not in message:
            raise RuntimeError("A `hello` message without the `can_act` key.")
        can_act = message["can_act"]
        if not can_act:
            raise RuntimeError(
                "A `hello` message with an invalid `can_act` "
                f"(can_act = {can_act})."
            )

        my_name = self._game_state.get_my_name()
        response = json.dumps(
            {"type": "join", "name": my_name, "room": "default"}
        )
        print(response, flush=True)

    def _on_start_game(self, message: dict) -> None:
        assert message["type"] == "start_game"

        seat = int(sys.argv[1])
        if seat < 0 or 4 <= seat:
            raise RuntimeError(f"{seat}: An invalid seat.")
        self._game_state.on_new_game(seat)

        response = json.dumps({"type": "none"})
        print(response, flush=True)

    def _on_start_kyoku(self, message: dict) -> None:
        assert message["type"] == "start_kyoku"

        seat = self._game_state.get_seat()

        if "bakaze" not in message:
            raise RuntimeError(
                "A `start_kyoku` message without the `bakaze` key."
            )
        _chang: str = message["bakaze"]
        if not isinstance(_chang, str):
            raise RuntimeError(type(_chang))
        if _chang not in ("E", "S", "W"):
            raise RuntimeError(
                "A `start_kyoku` message with an invalid `bakaze` "
                f"(bakaze = {_chang})."
            )
        chang = {"E": 0, "S": 1, "W": 2}[_chang]

        if "kyoku" not in message:
            raise RuntimeError(
                "A `start_kyoku` message without the `kyoku` key."
            )
        ju: int = message["kyoku"]
        if not isinstance(ju, int):
            raise RuntimeError(type(ju))
        if ju < 1 or 4 < ju:
            raise RuntimeError(
                "A `start_kyoku` message with an invalid `kyoku` "
                f"(kyoku = {ju})."
            )
        ju -= 1

        if "honba" not in message:
            raise RuntimeError(
                "A `start_kyoku` message without the `honba` key."
            )
        ben_chang: int = message["honba"]
        if not isinstance(ben_chang, int):
            raise RuntimeError(type(ben_chang))
        if ben_chang < 0:
            raise RuntimeError(
                "A `start_kyoku` message with an invalid `honba`"
                f" (honba = {ben_chang})."
            )

        if "kyotaku" not in message:
            raise RuntimeError(
                "A `start_kyoku` message without the `kyotaku` key."
            )
        deposits: int = message["kyotaku"]
        if not isinstance(deposits, int):
            raise RuntimeError(type(deposits))
        if deposits < 0:
            raise RuntimeError(
                "A `start_kyoku` message with an invalid `kyotaku` "
                f"(kyotaku = {deposits})."
            )

        if "oya" not in message:
            raise RuntimeError(
                "A `start_kyoku` message without the `oya` key."
            )
        dealer: int = message["oya"]
        if not isinstance(dealer, int):
            raise RuntimeError(type(dealer))
        if dealer != ju:
            raise RuntimeError(
                "An inconsistent `start_kyoku` message "
                f"(kyoku = {ju + 1}, oya = {dealer})."
            )

        if "dora_marker" not in message:
            raise RuntimeError(
                "A `start_kyoku` message without the `dora_marker` key."
            )
        _dora_indicator = message["dora_marker"]
        if _dora_indicator not in _TILE2NUM:
            raise RuntimeError(
                "A `start_kyoku` message with an invalid `dora_marker` "
                f"(dora_marker = {_dora_indicator})."
            )
        dora_indicator = _TILE2NUM[_dora_indicator]

        if "scores" not in message:
            raise RuntimeError(
                "A `start_kyoku` message without the `scores` key."
            )
        scores: list[int] = message["scores"]
        if not isinstance(scores, list):
            raise RuntimeError(type(scores))
        if len(scores) != 4:
            raise RuntimeError(
                f"A `start_kyoku` message with an invalid scores ({scores})."
            )
        for score in scores:
            if not isinstance(score, int):
                raise RuntimeError(type(score))

        if "tehais" not in message:
            raise RuntimeError(
                "A `start_kyoku` message without the `tehais` key."
            )
        hands: list[list[str]] = message["tehais"]
        if not isinstance(hands, list):
            raise RuntimeError(type(hands))
        if len(hands) != 4:
            raise RuntimeError(
                f"A `start_kyoku` message with an wrong `tehais` ({hands})."
            )
        for _hand in hands:
            if not isinstance(_hand, list):
                raise RuntimeError(type(_hand))
            if len(_hand) != 13:
                raise RuntimeError(len(_hand))
            for h in _hand:
                if h not in _TILE2NUM and h != "?":
                    raise RuntimeError(h)

        hand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i, _hand in enumerate(hands):
            if i != seat:
                if _hand != [
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                ]:
                    raise RuntimeError(
                        "A `start_kyoku` message with an wrong `tehais` "
                        f"(seat = {seat}, i = {i}, hand = {_hand})."
                    )
            else:
                if len(_hand) != 13:
                    raise RuntimeError(
                        "A `start_kyoku` message with an wrong `tehais` "
                        f"(seat = {seat}, hand = {_hand})."
                    )
                for i in range(13):
                    if _hand[i] not in _TILE2NUM:
                        raise RuntimeError(
                            "A `start_kyoku` message with an wrong `tehais` "
                            f"(seat = {seat}, hand = {_hand})."
                        )
                    hand[i] = _TILE2NUM[_hand[i]]

        self._game_state.on_new_round(scores)
        self._round_state.on_new_round(
            chang, ju, ben_chang, deposits, dora_indicator, hand
        )

    def _respond(self, dapai: int | None, candidates: list[int]) -> None:
        if dapai is not None and (dapai < 0 or dapai >= 37):
            raise ValueError(f"An invalid `dapai` value: {dapai}")
        if len(candidates) == 0:
            raise ValueError("An empty `candidates` list.")

        seat = self._game_state.get_seat()

        sparse = []
        sparse.append(self._game_state.get_room())
        sparse.append(self._game_state.get_game_style() + 5)
        sparse.append(self._game_state.get_player_grade(0) + 7)
        sparse.append(self._game_state.get_player_grade(1) + 23)
        sparse.append(self._game_state.get_player_grade(2) + 39)
        sparse.append(self._game_state.get_player_grade(3) + 55)
        sparse.append(seat + 71)
        sparse.append(self._round_state.get_chang() + 75)
        sparse.append(self._round_state.get_ju() + 78)
        sparse.append(self._round_state.get_num_left_tiles() + 82)
        for i, dora_indicator in enumerate(
            self._round_state.get_dora_indicators()
        ):
            sparse.append(dora_indicator + 37 * i + 152)
        hand_encode = [0 for _ in range(136)]
        for tile in self._round_state.get_my_hand():
            flag = False
            for i in range(_TILE_OFFSETS[tile], _TILE_OFFSETS[tile + 1]):
                if hand_encode[i] == 0:
                    hand_encode[i] = 1
                    flag = True
                    break
            if not flag:
                raise RuntimeError("TODO: (A suitable error message)")
        for i in range(136):
            if hand_encode[i] == 1:
                sparse.append(i + 337)
        zimo_tile = self._round_state.get_zimo_tile()
        if zimo_tile is not None:
            sparse.append(zimo_tile + 473)
        for i in range(len(sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
            sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
        _sparse = torch.tensor(sparse, device=self._device, dtype=torch.int32)
        _sparse = torch.unsqueeze(_sparse, dim=0)

        numeric = []
        numeric.append(self._round_state.get_num_ben_chang())
        numeric.append(self._round_state.get_num_deposits())
        numeric.append(self._game_state.get_player_score(0))
        numeric.append(self._game_state.get_player_score(1))
        numeric.append(self._game_state.get_player_score(2))
        numeric.append(self._game_state.get_player_score(3))
        numeric_ = torch.tensor(
            numeric, device=self._device, dtype=torch.int32
        )
        numeric_ = torch.unsqueeze(numeric_, dim=0)

        progression = self._round_state.copy_progression()
        for i in range(len(progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
            progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
        progression_ = torch.tensor(
            progression, device=self._device, dtype=torch.int32
        )
        progression_ = torch.unsqueeze(progression_, dim=0)

        candidates_ = list(candidates)
        for i in range(len(candidates_), MAX_NUM_ACTION_CANDIDATES):
            candidates_.append(NUM_TYPES_OF_ACTIONS)
        candidates__ = torch.tensor(
            candidates_, device=self._device, dtype=torch.int32
        )
        candidates__ = torch.unsqueeze(candidates__, dim=0)

        src = {
            "sparse": _sparse,
            "numeric": numeric_,
            "progression": progression_,
            "candidates": candidates__,
        }
        data = TensorDict(src, batch_size=1, device=self._device)
        with torch.no_grad():
            self._model(data)
        if data.get("action", None) is None:
            raise RuntimeError("TODO: (A suitable error message)")
        action_index = int(data["action"].squeeze().item())
        if action_index < 0:
            raise RuntimeError("TODO: (A suitable error message)")
        if action_index >= len(candidates):
            raise RuntimeError("TODO: (A suitable error message)")
        decision = candidates[action_index]

        if 0 <= decision and decision <= 147:
            tile = decision // 4
            _tile = _NUM2TILE[tile]
            encode = decision % 4
            moqi = encode // 2 == 1
            encode = encode % 2
            liqi = encode == 1

            if liqi:
                response = json.dumps({"type": "reach", "actor": seat})
                print(response, flush=True)
                messages = sys.stdin.readline()
                messages = json.loads(messages)
                if len(messages) > 1:
                    raise RuntimeError(
                        "Too many messages starting with `reach`."
                    )
                message = messages[0]
                if "type" not in message:
                    raise RuntimeError("A message without the `type` key.")
                message_type: str = message["type"]
                if not isinstance(message_type, str):
                    raise RuntimeError(
                        "A message with an invalid `type` "
                        f"({type(message_type)})."
                    )
                if message_type != "reach":
                    raise RuntimeError(
                        "A `reach` message is expected, "
                        f"but got a `{message_type}` message"
                    )
                if "actor" not in message:
                    raise RuntimeError(
                        "A `reach` message without the `actor` key."
                    )
                actor = message["actor"]
                if not isinstance(actor, int):
                    raise RuntimeError(
                        "A `reach` message with an invalid `actor` "
                        f"({type(actor)})."
                    )
                if actor != seat:
                    raise RuntimeError(
                        "A `reach` message with an invalid actor "
                        f"(actor = {actor})."
                    )
                self._round_state.on_liqi(seat)

            response = json.dumps(
                {
                    "type": "dahai",
                    "actor": seat,
                    "pai": _tile,
                    "tsumogiri": moqi,
                }
            )
            print(response, flush=True)
            return

        if 148 <= decision and decision <= 181:
            angang = _NUM2ANGANG[decision - 148]
            response = json.dumps(
                {"type": "ankan", "actor": seat, "consumed": angang}
            )
            print(response, flush=True)
            return

        if 182 <= decision and decision <= 218:
            _tile, consumed = _NUM2JIAGANG[decision - 182]
            response = json.dumps(
                {
                    "type": "kakan",
                    "actor": seat,
                    "pai": _tile,
                    "consumed": consumed,
                }
            )
            print(response, flush=True)
            return

        if decision == 219:
            hupai = self._round_state.get_zimo_tile()
            if hupai is None:
                raise RuntimeError("Trying zimohu without any zimo tile.")
            _hupai = _NUM2TILE[hupai]
            response = json.dumps(
                {"type": "hora", "actor": seat, "target": seat, "pai": _hupai}
            )
            print(response, flush=True)
            return

        if decision == 220:
            response = json.dumps({"type": "ryukyoku"})
            print(response, flush=True)
            return

        if decision == 221:
            response = json.dumps({"type": "none"})
            print(response, flush=True)
            in_liqi = self._round_state.is_in_liqi()
            for i in (543, 544, 545):
                if i in candidates:
                    # 栄和が選択肢にあるにも関わらず見逃しを選択した．
                    # この結果，フリテンが発生する．
                    self._round_state.set_zhenting(2 if in_liqi else 1)
                    break
            return

        if 222 <= decision and decision <= 311:
            _tile, consumed = _NUM2CHI[decision - 222]
            response = json.dumps(
                {
                    "type": "chi",
                    "actor": seat,
                    "target": (seat + 3) % 4,
                    "pai": _tile,
                    "consumed": consumed,
                }
            )
            print(response, flush=True)
            return

        if 312 <= decision and decision <= 431:
            encode = decision - 312
            relseat = encode // 40
            target = (seat + relseat + 1) % 4
            encode = encode % 40
            _tile, consumed = _NUM2PENG[encode]
            response = json.dumps(
                {
                    "type": "pon",
                    "actor": seat,
                    "target": target,
                    "pai": _tile,
                    "consumed": consumed,
                }
            )
            print(response, flush=True)
            return

        if 432 <= decision and decision <= 542:
            encode = decision - 432
            relseat = encode // 37
            target = (seat + relseat + 1) % 4
            encode = encode % 37
            _tile, consumed = _NUM2DAMINGGANG[encode]
            response = json.dumps(
                {
                    "type": "daiminkan",
                    "actor": seat,
                    "target": target,
                    "pai": _tile,
                    "consumed": consumed,
                }
            )
            print(response, flush=True)
            return

        if 543 <= decision and decision <= 545:
            relseat = decision - 543
            target = (seat + relseat + 1) % 4
            hupai = dapai
            if hupai is None:
                raise RuntimeError("Trying rong without any dapai.")
            _hupai = _NUM2TILE[hupai]
            response = json.dumps(
                {
                    "type": "hora",
                    "actor": seat,
                    "target": target,
                    "pai": _hupai,
                }
            )
            print(response, flush=True)
            return

        raise RuntimeError(f"An invalid decision (decision = {decision}).")

    def _on_zimo(self, message: dict) -> None:
        assert message["type"] == "tsumo"

        seat = self._game_state.get_seat()

        if "actor" not in message:
            raise RuntimeError("A `tsumo` message without the `actor` key.")
        actor = message["actor"]
        if not isinstance(actor, int):
            raise RuntimeError(type(actor))
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f"A `tsumo` message with an invalid `actor` (actor = {actor})."
            )
        mine = actor == seat

        if "pai" not in message:
            raise RuntimeError("A `tsumo` message without the `pai` key.")
        tile = message["pai"]
        if not isinstance(tile, str):
            raise RuntimeError(type(tile))

        my_score = self._game_state.get_player_score(seat)

        if not mine:
            if tile != "?":
                raise RuntimeError(
                    "An inconsistent `tsumo` message "
                    f"(seat = {seat}, actor = {actor}, pai = {tile})."
                )
            self._round_state.on_zimo(seat, mine, None, my_score)
        else:
            if tile not in _TILE2NUM:
                raise RuntimeError(
                    "A `tsumo` message with an invalid `pai` "
                    f"(seat = {seat}, actor = {actor}, pai = {tile})."
                )
            _tile = _TILE2NUM[tile]
            candidates = self._round_state.on_zimo(seat, mine, _tile, my_score)
            if len(candidates) == 0:
                raise RuntimeError("The length of `candidates` is equal to 0.")
            self._respond(None, candidates)

    def _on_dapai(self, message: dict) -> None:
        assert message["type"] == "dahai"

        seat = self._game_state.get_seat()

        if "actor" not in message:
            raise RuntimeError("A `dahai` message without the `actor` key.")
        actor = message["actor"]
        if not isinstance(actor, int):
            raise RuntimeError(type(actor))
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f"A `dahai` message with an invalid `actor` (actor = {actor})."
            )

        if "pai" not in message:
            raise RuntimeError("A `dahai` message without the `pai` key.")
        tile = message["pai"]
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f"A `dahai` message with an invalid `pai` (pai = {tile})."
            )
        _tile = _TILE2NUM[tile]

        if "tsumogiri" not in message:
            raise RuntimeError(
                "A `dahai` message without the `tsumogiri` key."
            )
        moqi = message["tsumogiri"]
        if not isinstance(moqi, bool):
            raise RuntimeError(type(moqi))

        candidates = self._round_state.on_dapai(seat, actor, _tile, moqi)

        if actor == seat:
            # 自身の打牌に対してやることは何もない．
            if len(candidates) >= 1:
                raise RuntimeError(candidates)
            return

        if len(candidates) == 0:
            return

        if len(candidates) < 2:
            raise RuntimeError(candidates)
        self._respond(_tile, candidates)

    def _on_chi(self, message: dict) -> None:
        assert message["type"] == "chi"

        if "actor" not in message:
            raise RuntimeError("A `chi` message without the `actor` key.")
        actor = message["actor"]
        if not isinstance(actor, int):
            raise RuntimeError(type(actor))
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f"A `chi` message with an invalid `actor` (actor = {actor})."
            )
        mine = actor == self._game_state.get_seat()

        if "target" not in message:
            raise RuntimeError("A `chi` message without the `target` key.")
        target = message["target"]
        if not isinstance(target, int):
            raise RuntimeError(type(target))
        if target < 0 or 4 <= target:
            raise RuntimeError(
                "A `chi` message with an invalid `target` "
                f"(target = {target})."
            )
        if (target + 4 - actor) % 4 != 3:
            raise RuntimeError(
                "An inconsistent `chi` message "
                f"(actor = {actor}, target = {target})."
            )

        if "pai" not in message:
            raise RuntimeError("A `chi` message without the `pai` key.")
        tile = message["pai"]
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f"A `chi` message with an invalid `pai` (pai = {tile})."
            )

        if "consumed" not in message:
            raise RuntimeError("A `pon` message without the `consumed` key.")
        consumed: list[str] = message["consumed"]
        if len(consumed) != 2:
            raise RuntimeError(
                "A `chi` message with an invalid `consumed` "
                f"(consumed = {consumed})."
            )
        for t in consumed:
            if t not in _TILE2NUM:
                raise RuntimeError(
                    "A `chi` message with an invalid `consumed` "
                    f"(consumed = {consumed})."
                )

        chi = (tile, (consumed[0], consumed[1]))
        if chi not in _CHI2NUM:
            raise RuntimeError(chi)
        _chi = _CHI2NUM[chi]

        candidates = self._round_state.on_chi(mine, actor, _chi)

        if len(candidates) == 0:
            if mine:
                # 自身のチーの直後の場合，必ず打牌の選択肢が存在するはずである．
                raise RuntimeError("TODO: (A suitable error message)")
            return

        if not mine:
            # 他家のチーの直後の場合，いかなる選択肢も存在してはならない．
            raise RuntimeError("TODO: (A suitable error message)")
        self._respond(None, candidates)

    def _on_peng(self, message: dict) -> None:
        assert message["type"] == "pon"

        if "actor" not in message:
            raise RuntimeError("A `pon` message without the `actor` key.")
        actor = message["actor"]
        if not isinstance(actor, int):
            raise RuntimeError(type(actor))
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f"A `pon` message with an invalid `actor` (actor = {actor})."
            )
        mine = actor == self._game_state.get_seat()

        if "target" not in message:
            raise RuntimeError("A `pon` message without the `target` key.")
        target = message["target"]
        if not isinstance(target, int):
            raise RuntimeError(type(target))
        if target < 0 or 4 <= target:
            raise RuntimeError(
                f"A `pon` message with an invalid `target` (target = {target})."
            )
        if actor == target:
            raise RuntimeError(
                "An inconsistent `pon` message "
                f"(actor = {actor}, target = {target})."
            )
        relseat = (target + 4 - actor) % 4 - 1

        if "pai" not in message:
            raise RuntimeError("A `pon` message without the `pai` key.")
        tile = message["pai"]
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f"A `pon` message with an invalid `pai` (pai = {tile})."
            )

        if "consumed" not in message:
            raise RuntimeError("A `pon` message without the `consumed` key.")
        consumed: list[str] = message["consumed"]
        if len(consumed) != 2:
            raise RuntimeError(
                "A `pon` message with an invalid `consumed` "
                f"(consumed = {consumed})."
            )
        for t in consumed:
            if t not in _TILE2NUM:
                raise RuntimeError(
                    "A `pon` message with an invalid `consumed` "
                    f"(consumed = {consumed})."
                )

        peng = (tile, (consumed[0], consumed[1]))
        if peng not in _PENG2NUM:
            raise RuntimeError(peng)
        _peng = _PENG2NUM[peng]

        candidates = self._round_state.on_peng(mine, actor, relseat, _peng)

        if len(candidates) == 0:
            if mine:
                # 自身のポンの直後の場合，必ず打牌の選択肢が存在するはずである．
                raise RuntimeError("TODO: (A suitable error message)")
            return

        if not mine:
            # 他家のポンの直後の場合，いかなる選択肢も存在してはならない．
            raise RuntimeError("TODO: (A suitable error message)")
        self._respond(None, candidates)

    def _on_daminggang(self, message: dict) -> None:
        assert message["type"] == "daiminkan"

        if "actor" not in message:
            raise RuntimeError(
                "A `daiminkan` message without the `actor` key."
            )
        actor = message["actor"]
        if not isinstance(actor, int):
            raise RuntimeError(type(actor))
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                "A `daiminkan` message with an invalid `actor` "
                f"(actor = {actor})."
            )
        mine = actor == self._game_state.get_seat()

        if "target" not in message:
            raise RuntimeError(
                "A `daiminkan` message without the `target` key."
            )
        target = message["target"]
        if not isinstance(target, int):
            raise RuntimeError(type(target))
        if target < 0 or 4 <= target:
            raise RuntimeError(
                "A `daiminkan` message with an invalid `target` "
                f"(target = {target})."
            )
        if actor == target:
            raise RuntimeError(
                "An inconsistent `daiminkan` message "
                f"(actor = {actor}, target = {target})."
            )
        relseat = (target + 4 - actor) % 4 - 1

        if "pai" not in message:
            raise RuntimeError("A `daiminkan` message without the `pai` key.")
        tile = message["pai"]
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f"A `daiminkan` message with an invalid `pai` (pai = {tile})."
            )

        if "consumed" not in message:
            raise RuntimeError(
                "A `daiminkan` message without the `consumed` key."
            )
        consumed: list[str] = message["consumed"]
        if len(consumed) != 3:
            raise RuntimeError(
                "A `daiminkan` message with an invalid `consumed` "
                f"(consumed = {consumed})."
            )
        for t in consumed:
            if t not in _TILE2NUM:
                raise RuntimeError(
                    "A `daiminkan` message with an invalid `consumed` "
                    f"(consumed = {consumed})."
                )

        daminggang = (tile, (consumed[0], consumed[1], consumed[2]))
        if daminggang not in _DAMINGGANG2NUM:
            raise RuntimeError(daminggang)
        _daminggang = _DAMINGGANG2NUM[daminggang]

        self._round_state.on_daminggang(mine, actor, relseat, _daminggang)

    def _on_angang(self, message: dict) -> None:
        assert message["type"] == "ankan"

        seat = self._game_state.get_seat()

        if "actor" not in message:
            raise RuntimeError("A `ankan` message without the `actor` key.")
        actor = message["actor"]
        if not isinstance(actor, int):
            raise RuntimeError(type(actor))
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f"A `ankan` message with an invalid `actor` (actor = {actor})."
            )
        mine = actor == seat

        if "consumed" not in message:
            raise RuntimeError("A `ankan` message without the `consumed` key.")
        consumed: list[str] = message["consumed"]
        if len(consumed) != 4:
            raise RuntimeError(
                "A `ankan` message with an invalid `consumed` "
                f"(consumed = {consumed})."
            )
        for t in consumed:
            if t not in _TILE2NUM:
                raise RuntimeError(
                    "A `ankan` message with an invalid `consumed` "
                    f"(consumed = {consumed})."
                )
        angang = (consumed[0], consumed[1], consumed[2], consumed[3])

        if angang not in _ANGANG2NUM:
            raise RuntimeError(angang)
        _angang = _ANGANG2NUM[angang]

        candidates = self._round_state.on_angang(seat, actor, _angang)
        if mine:
            if len(candidates) >= 1:
                # 自身の暗槓の直後の場合，いかなる選択肢も存在してはならない．
                raise RuntimeError(candidates)
            return

        if len(candidates) == 0:
            return

        if len(candidates) != 2:
            raise RuntimeError(len(candidates))
        # 国士無双の槍槓の可能性があるため，暗槓は打牌とみなす．
        dapai = _TILE2NUM[consumed[0]]
        self._respond(dapai, candidates)

    def _on_jiagang(self, message: dict) -> None:
        assert message["type"] == "kakan"

        seat = self._game_state.get_seat()

        if "actor" not in message:
            raise RuntimeError("A `kakan` message without the `actor` key.")
        actor = message["actor"]
        if not isinstance(actor, int):
            raise RuntimeError(type(actor))
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f"A `kakan` message with an invalid `actor` (actor = {actor})."
            )
        mine = actor == seat

        if "pai" not in message:
            raise RuntimeError("A `kakan` message without the `pai` key.")
        tile = message["pai"]
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f"A `kakan` message with an invalid `pai` (pai = {tile})."
            )
        _tile = _TILE2NUM[tile]

        if "consumed" not in message:
            raise RuntimeError("A `kakan` message without the `consumed` key.")
        consumed: list[str] = message["consumed"]
        if len(consumed) != 3:
            raise RuntimeError(
                "A `kakan` message with an invalid `consumed` "
                f"(consumed = {consumed})."
            )
        for t in consumed:
            if t not in _TILE2NUM:
                raise RuntimeError(
                    "A `kakan` message with an invalid `consumed` "
                    f"(consumed = {consumed})."
                )

        candidates = self._round_state.on_jiagang(seat, actor, _tile)
        if mine:
            if len(candidates) >= 1:
                # 自身の加槓の直後の場合，いかなる選択肢も存在してはならない．
                raise RuntimeError(candidates)
            return

        if len(candidates) == 0:
            return

        # 槍槓の選択肢が発生している．
        if len(candidates) != 2:
            raise RuntimeError(len(candidates))
        self._respond(_tile, candidates)

    def _on_liqi(self, message: dict) -> None:
        assert message["type"] == "reach"

        if "actor" not in message:
            raise RuntimeError("A `reach` message without the `actor` key.")
        actor = message["actor"]
        if not isinstance(actor, int):
            raise RuntimeError(type(actor))
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f"A `reach` message with an invalid `actor` (actor = {actor})."
            )

        self._round_state.on_liqi(actor)

    def _on_liqi_acceptance(self, message: dict) -> None:
        assert message["type"] == "reach_accepted"

        if "actor" not in message:
            raise RuntimeError(
                "A `reach_accepted` message without the `actor` key."
            )
        actor = message["actor"]
        if not isinstance(actor, int):
            raise RuntimeError(type(actor))
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f"A `reach_accepted` message with an invalid `actor` (actor = {actor})."
            )
        mine = actor == self._game_state.get_seat()

        self._game_state.on_liqi_acceptance(actor)
        self._round_state.on_liqi_acceptance(mine, actor)

    def _on_new_dora(self, message: dict) -> None:
        assert message["type"] == "dora"

        if "dora_marker" not in message:
            raise RuntimeError(
                "A `dora` message without the `dora_marker` key."
            )
        dora_indicator = message["dora_marker"]
        if dora_indicator not in _TILE2NUM:
            raise RuntimeError(
                "A `dora` message with an invalid `dora_marker` "
                f"(dora_marker = {dora_indicator})."
            )
        _dora_indicator = _TILE2NUM[dora_indicator]

        self._round_state.on_new_dora(_dora_indicator)

    def _on_hulu(self, message: dict) -> None:
        assert message["type"] == "hora"

        if "actor" not in message:
            raise RuntimeError("A `hora` message without the `actor` key.")
        actor = message["actor"]
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f"A `hora` message with an invalid `actor` (actor = {actor})."
            )

        if "target" not in message:
            raise RuntimeError("A `hora` message without the `target` key.")
        target = message["target"]
        if target < 0 or 4 <= target:
            raise RuntimeError(
                f"A `hora` message with an invalid `target` (target = {target})."
            )

        if "pai" not in message:
            raise RuntimeError("A `hora` message without the `pai` key.")
        tile = message["pai"]
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f"A `hora` message with an invalid `pai` (pai = {tile})."
            )

    def _on_luju(self, message: dict) -> None:
        assert message["type"] == "ryukyoku"

        if "can_act" not in message:
            raise RuntimeError(
                "A `ryukyoku` message without the `can_act` key."
            )
        can_act = message["can_act"]
        if not isinstance(can_act, bool):
            raise RuntimeError(type(can_act))
        if can_act:
            raise RuntimeError(
                f"An inconsistent `ryukyoku` message (can_act = {can_act})."
            )

    def _on_round_end(self, message: dict) -> None:
        assert message["type"] == "end_kyoku"

        response = json.dumps({"type": "none"})
        print(response, flush=True)

    def _on_game_end(self, message: dict) -> None:
        assert message["type"] == "end_game"

        response = json.dumps({"type": "none"})
        print(response, flush=True)

    def run(self) -> None:
        messages: list[dict[str, Any]] = []
        while True:
            if len(messages) == 0:
                message_line = sys.stdin.readline()
                if message_line.strip() == "":
                    # WORKAROUND
                    continue
                messages = json.loads(message_line)
                if len(messages) == 0:
                    raise RuntimeError("The standard input is empty.")

            message = messages[0]
            if "type" not in message:
                raise RuntimeError("A message without the `type` key.")

            if message["type"] == "hello":
                if len(messages) > 1:
                    raise RuntimeError("A multi-line `hello` message.")
                self._on_hello(message)
                messages.pop(0)
                continue

            if message["type"] == "start_game":
                if len(messages) != 1:
                    raise RuntimeError(
                        "Too many messages starting with `start_game`."
                    )
                self._on_start_game(message)
                messages.pop(0)
                continue

            if message["type"] == "start_kyoku":
                if len(messages) < 2:
                    raise RuntimeError(
                        "Too few messages starting with `start_kyoku`."
                    )
                self._on_start_kyoku(message)
                messages.pop(0)
                continue

            if message["type"] == "tsumo":
                self._on_zimo(message)
                messages.pop(0)
                continue

            if message["type"] == "dahai":
                self._on_dapai(message)
                messages.pop(0)
                continue

            if message["type"] == "chi":
                self._on_chi(message)
                messages.pop(0)
                continue

            if message["type"] == "pon":
                self._on_peng(message)
                messages.pop(0)
                continue

            if message["type"] == "daiminkan":
                self._on_daminggang(message)
                messages.pop(0)
                continue

            if message["type"] == "ankan":
                self._on_angang(message)
                messages.pop(0)
                continue

            if message["type"] == "kakan":
                self._on_jiagang(message)
                messages.pop(0)
                continue

            if message["type"] == "reach":
                self._on_liqi(message)
                messages.pop(0)
                continue

            if message["type"] == "reach_accepted":
                self._on_liqi_acceptance(message)
                messages.pop(0)
                continue

            if message["type"] == "dora":
                self._on_new_dora(message)
                messages.pop(0)
                continue

            if message["type"] == "hora":
                self._on_hulu(message)
                messages.pop(0)
                continue

            if message["type"] == "ryukyoku":
                self._on_luju(message)
                messages.pop(0)
                continue

            if message["type"] == "end_kyoku":
                self._on_round_end(message)
                messages.pop(0)
                continue

            if message["type"] == "end_game":
                self._on_game_end(message)
                messages.pop(0)
                if len(messages) > 0:
                    raise RuntimeError("TODO: (A suitable error message)")
                continue

            raise RuntimeError(message)
