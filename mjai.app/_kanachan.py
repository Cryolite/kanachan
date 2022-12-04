#!/usr/bin/env python3

import re
from pathlib import Path
from collections import Counter
import json
import sys
from typing import (Optional, List,)
import torch
from torch import nn
from xiangting_calculator import XiangtingCalculator
from hand_calculator import HandCalculator
from kanachan.training.common import load_state_dict
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES)
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.policy_model import (PolicyDecoder, PolicyModel,)


_NUM2TILE = (
    '5mr', '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
    '5pr', '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
    '5sr', '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
    'E', 'S', 'W', 'N', 'P', 'F', 'C'
)


_TILE2NUM = {}
for i, tile in enumerate(_NUM2TILE):
    _TILE2NUM[tile] = i


_TILE_OFFSETS = (
      0,   1,   5,   9,  13,  17,  20,  24,  28,  32,
     36,  37,  41,  45,  49,  53,  56,  60,  64,  68,
     72,  73,  77,  81,  85,  89,  92,  96, 100, 104,
    108, 112, 116, 120, 124, 128, 132, 136
)


_NUM2CHI = (
    ('1m',  ['2m', '3m']),
    ('2m',  ['1m', '3m']),
    ('2m',  ['3m', '4m']),
    ('3m',  ['1m', '2m']),
    ('3m',  ['2m', '4m']),
    ('3m',  ['4m', '5m']),
    ('3m',  ['4m', '5mr']),
    ('4m',  ['2m', '3m']),
    ('4m',  ['3m', '5m']),
    ('4m',  ['3m', '5mr']),
    ('4m',  ['5m', '6m']),
    ('4m',  ['5mr', '6m']),
    ('5m',  ['3m', '4m']),
    ('5mr', ['3m', '4m']),
    ('5m',  ['4m', '6m']),
    ('5mr', ['4m', '6m']),
    ('5m',  ['6m', '7m']),
    ('5mr', ['6m', '7m']),
    ('6m',  ['4m', '5m']),
    ('6m',  ['4m', '5mr']),
    ('6m',  ['5m', '7m']),
    ('6m',  ['5mr', '7m']),
    ('6m',  ['7m', '8m']),
    ('7m',  ['5m', '6m']),
    ('7m',  ['5mr', '6m']),
    ('7m',  ['6m', '8m']),
    ('7m',  ['8m', '9m']),
    ('8m',  ['6m', '7m']),
    ('8m',  ['7m', '9m']),
    ('9m',  ['7m', '8m']),
    ('1p',  ['2p', '3p']),
    ('2p',  ['1p', '3p']),
    ('2p',  ['3p', '4p']),
    ('3p',  ['1p', '2p']),
    ('3p',  ['2p', '4p']),
    ('3p',  ['4p', '5p']),
    ('3p',  ['4p', '5pr']),
    ('4p',  ['2p', '3p']),
    ('4p',  ['3p', '5p']),
    ('4p',  ['3p', '5pr']),
    ('4p',  ['5p', '6p']),
    ('4p',  ['5pr', '6p']),
    ('5p',  ['3p', '4p']),
    ('5pr', ['3p', '4p']),
    ('5p',  ['4p', '6p']),
    ('5pr', ['4p', '6p']),
    ('5p',  ['6p', '7p']),
    ('5pr', ['6p', '7p']),
    ('6p',  ['4p', '5p']),
    ('6p',  ['4p', '5pr']),
    ('6p',  ['5p', '7p']),
    ('6p',  ['5pr', '7p']),
    ('6p',  ['7p', '8p']),
    ('7p',  ['5p', '6p']),
    ('7p',  ['5pr', '6p']),
    ('7p',  ['6p', '8p']),
    ('7p',  ['8p', '9p']),
    ('8p',  ['6p', '7p']),
    ('8p',  ['7p', '9p']),
    ('9p',  ['7p', '8p']),
    ('1s',  ['2s', '3s']),
    ('2s',  ['1s', '3s']),
    ('2s',  ['3s', '4s']),
    ('3s',  ['1s', '2s']),
    ('3s',  ['2s', '4s']),
    ('3s',  ['4s', '5s']),
    ('3s',  ['4s', '5sr']),
    ('4s',  ['2s', '3s']),
    ('4s',  ['3s', '5s']),
    ('4s',  ['3s', '5sr']),
    ('4s',  ['5s', '6s']),
    ('4s',  ['5sr', '6s']),
    ('5s',  ['3s', '4s']),
    ('5sr', ['3s', '4s']),
    ('5s',  ['4s', '6s']),
    ('5sr', ['4s', '6s']),
    ('5s',  ['6s', '7s']),
    ('5sr', ['6s', '7s']),
    ('6s',  ['4s', '5s']),
    ('6s',  ['4s', '5sr']),
    ('6s',  ['5s', '7s']),
    ('6s',  ['5sr', '7s']),
    ('6s',  ['7s', '8s']),
    ('7s',  ['5s', '6s']),
    ('7s',  ['5sr', '6s']),
    ('7s',  ['6s', '8s']),
    ('7s',  ['8s', '9s']),
    ('8s',  ['6s', '7s']),
    ('8s',  ['7s', '9s']),
    ('9s',  ['7s', '8s'])
)


_CHI2NUM = {}
for i, (tile, consumed) in enumerate(_NUM2CHI):
    k = (tile, tuple(consumed))
    _CHI2NUM[k] = i


_CHI_COUNTS = []
for tile, consumed in _NUM2CHI:
    tile = _TILE2NUM[tile]
    counts = {}
    for t in consumed:
        t = _TILE2NUM[t]
        if t not in counts:
            counts[t] = 0
        counts[t] += 1
    _CHI_COUNTS.append((tile, counts))


_CHI_TO_KUIKAE_TILES = (
    ( 1,  4,),     # (2m, 3m, 1m) => 1m, 4m
    ( 2,),         # (1m, 3m, 2m) => 2m
    ( 2,  0,  5,), # (3m, 4m, 2m) => 2m, 0m, 5m
    ( 3,),         # (1m, 2m, 3m) => 3m
    ( 3,),         # (2m, 4m, 3m) => 3m
    ( 3,  6,),     # (4m, 5m, 3m) => 3m, 6m
    ( 3,  6,),     # (4m, 0m, 3m) => 3m, 6m
    ( 1,  4,),     # (2m, 3m, 4m) => 1m, 4m
    ( 4,),         # (3m, 5m, 4m) => 4m
    ( 4,),         # (3m, 0m, 4m) => 4m
    ( 4,  7,),     # (5m, 6m, 4m) => 4m, 7m
    ( 4,  7,),     # (0m, 6m, 4m) => 4m, 7m
    ( 0,  2,  5,), # (3m, 4m, 5m) => 2m, 0m, 5m
    ( 2,  5,),     # (3m, 4m, 0m) => 2m, 5m
    ( 0,  5,),     # (4m, 6m, 5m) => 0m, 5m
    ( 5,),         # (4m, 6m, 0m) => 5m
    ( 0,  5,  8,), # (6m, 7m, 5m) => 0m, 5m, 8m
    ( 5,  8,),     # (6m, 7m, 0m) => 5m, 8m
    ( 3,  6,),     # (4m, 5m, 6m) => 3m, 6m
    ( 3,  6,),     # (4m, 0m, 6m) => 3m, 6m
    ( 6,),         # (5m, 7m, 6m) => 6m
    ( 6,),         # (0m, 7m, 6m) => 6m
    ( 6,  9,),     # (7m, 8m, 6m) => 6m, 9m
    ( 4,  7,),     # (5m, 6m, 7m) => 4m, 7m
    ( 4,  7,),     # (0m, 6m, 7m) => 4m, 7m
    ( 7,),         # (6m, 8m, 7m) => 7m
    ( 7,),         # (8m, 9m, 7m) => 7m
    ( 0,  5,  8,), # (6m, 7m, 8m) => 0m, 5m, 8m
    ( 8,),         # (7m, 9m, 8m) => 8m
    ( 6,  9,),     # (7m, 8m, 9m) => 9m
    (11, 14,),     # (2p, 3p, 1p) => 1p, 4p
    (12,),         # (1p, 3p, 2p) => 2p
    (12, 10, 15,), # (3p, 4p, 2p) => 2p, 0p, 5p
    (13,),         # (1p, 2p, 3p) => 3p
    (13,),         # (2p, 4p, 3p) => 3p
    (13, 16,),     # (4p, 5p, 3p) => 3p, 6p
    (13, 16,),     # (4p, 0p, 3p) => 3p, 6p
    (11, 14,),     # (2p, 3p, 4p) => 1p, 4p
    (14,),         # (3p, 5p, 4p) => 4p
    (14,),         # (3p, 0p, 4p) => 4p
    (14, 17,),     # (5p, 6p, 4p) => 4p, 7p
    (14, 17,),     # (0p, 6p, 4p) => 4p, 7p
    (10, 12, 15,), # (3p, 4p, 5p) => 2p, 0p, 5p
    (12, 15,),     # (3p, 4p, 0p) => 2p, 5p
    (10, 15,),     # (4p, 6p, 5p) => 0p, 5p
    (15,),         # (4p, 6p, 0p) => 5p
    (10, 15, 18,), # (6p, 7p, 5p) => 0p, 5p, 8p
    (15, 18,),     # (6p, 7p, 0p) => 5p, 8p
    (13, 16,),     # (4p, 5p, 6p) => 3p, 6p
    (13, 16,),     # (4p, 0p, 6p) => 3p, 6p
    (16,),         # (5p, 7p, 6p) => 6p
    (16,),         # (0p, 7p, 6p) => 6p
    (16, 19,),     # (7p, 8p, 6p) => 6p, 9p
    (14, 17,),     # (5p, 6p, 7p) => 4p, 7p
    (14, 17,),     # (0p, 6p, 7p) => 4p, 7p
    (17,),         # (6p, 8p, 7p) => 7p
    (17,),         # (8p, 9p, 7p) => 7p
    (10, 15, 18,), # (6p, 7p, 8p) => 0p, 5p, 8p
    (18,),         # (7p, 9p, 8p) => 8p
    (16, 19,),     # (7p, 8p, 9p) => 9p
    (21, 24,),     # (2s, 3s, 1s) => 1s, 4s
    (22,),         # (1s, 3s, 2s) => 2s
    (22, 20, 25,), # (3s, 4s, 2s) => 2s, 0s, 5s
    (23,),         # (1s, 2s, 3s) => 3s
    (23,),         # (2s, 4s, 3s) => 3s
    (23, 26,),     # (4s, 5s, 3s) => 3s, 6s
    (23, 26,),     # (4s, 0s, 3s) => 3s, 6s
    (21, 24,),     # (2s, 3s, 4s) => 1s, 4s
    (24,),         # (3s, 5s, 4s) => 4s
    (24,),         # (3s, 0s, 4s) => 4s
    (24, 27,),     # (5s, 6s, 4s) => 4s, 7s
    (24, 27,),     # (0s, 6s, 4s) => 4s, 7s
    (20, 22, 25,), # (3s, 4s, 5s) => 2s, 0s, 5s
    (22, 25,),     # (3s, 4s, 0s) => 2s, 5s
    (20, 25,),     # (4s, 6s, 5s) => 0s, 5s
    (25,),         # (4s, 6s, 0s) => 5s
    (20, 25, 28,), # (6s, 7s, 5s) => 0s, 5s, 8s
    (25, 28,),     # (6s, 7s, 0s) => 5s, 8s
    (23, 26,),     # (4s, 5s, 6s) => 3s, 6s
    (23, 26,),     # (4s, 0s, 6s) => 3s, 6s
    (26,),         # (5s, 7s, 6s) => 6s
    (26,),         # (0s, 7s, 6s) => 6s
    (26, 29,),     # (7s, 8s, 6s) => 6s, 9s
    (24, 27,),     # (5s, 6s, 7s) => 4s, 7s
    (24, 27,),     # (0s, 6s, 7s) => 4s, 7s
    (27,),         # (6s, 8s, 7s) => 7s
    (27,),         # (8s, 9s, 7s) => 7s
    (20, 25, 28,), # (6s, 7s, 8s) => 0s, 5s, 8s
    (28,),         # (7s, 9s, 8s) => 8s
    (26, 29,)      # (7s, 8s, 9s) => 9s
)


_NUM2PENG = (
    ('1m',  ['1m', '1m']),
    ('2m',  ['2m', '2m']),
    ('3m',  ['3m', '3m']),
    ('4m',  ['4m', '4m']),
    ('5m',  ['5m', '5m']),
    ('5m',  ['5mr', '5m']),
    ('5mr', ['5m', '5m']),
    ('6m',  ['6m', '6m']),
    ('7m',  ['7m', '7m']),
    ('8m',  ['8m', '8m']),
    ('9m',  ['9m', '9m']),
    ('1p',  ['1p', '1p']),
    ('2p',  ['2p', '2p']),
    ('3p',  ['3p', '3p']),
    ('4p',  ['4p', '4p']),
    ('5p',  ['5p', '5p']),
    ('5p',  ['5pr', '5p']),
    ('5pr', ['5p', '5p']),
    ('6p',  ['6p', '6p']),
    ('7p',  ['7p', '7p']),
    ('8p',  ['8p', '8p']),
    ('9p',  ['9p', '9p']),
    ('1s',  ['1s', '1s']),
    ('2s',  ['2s', '2s']),
    ('3s',  ['3s', '3s']),
    ('4s',  ['4s', '4s']),
    ('5s',  ['5s', '5s']),
    ('5s',  ['5sr', '5s']),
    ('5sr', ['5s', '5s']),
    ('6s',  ['6s', '6s']),
    ('7s',  ['7s', '7s']),
    ('8s',  ['8s', '8s']),
    ('9s',  ['9s', '9s']),
    ('E',   ['E', 'E']),
    ('S',   ['S', 'S']),
    ('W',   ['W', 'W']),
    ('N',   ['N', 'N']),
    ('P',   ['P', 'P']),
    ('F',   ['F', 'F']),
    ('C',   ['C', 'C'])
)


_PENG2NUM = {}
for i, (tile, consumed) in enumerate(_NUM2PENG):
    k = (tile, tuple(consumed))
    _PENG2NUM[k] = i


_PENG_COUNTS = []
for tile, consumed in _NUM2PENG:
    tile = _TILE2NUM[tile]
    counts = {}
    for t in consumed:
        t = _TILE2NUM[t]
        if t not in counts:
            counts[t] = 0
        counts[t] += 1
    _PENG_COUNTS.append((tile, counts))


_PENG_TO_KUIKAE_TILE = (
     1, # (1m, 1m, 1m) => 1m
     2, # (2m, 2m, 2m) => 2m
     3, # (3m, 3m, 3m) => 3m
     4, # (4m, 4m, 4m) => 4m
     0, # (5m, 5m, 5m) => 0m
     5, # (0m, 5m, 5m) => 5m
     5, # (5m, 5m, 0m) => 5m
     6, # (6m, 6m, 6m) => 6m
     7, # (7m, 7m, 7m) => 7m
     8, # (8m, 8m, 8m) => 8m
     9, # (9m, 9m, 9m) => 9m
    11, # (1p, 1p, 1p) => 1p
    12, # (2p, 2p, 2p) => 2p
    13, # (3p, 3p, 3p) => 3p
    14, # (4p, 4p, 4p) => 4p
    10, # (5p, 5p, 5p) => 0p
    15, # (0p, 5p, 5p) => 5p
    15, # (5p, 5p, 0p) => 5p
    16, # (6p, 6p, 6p) => 6p
    17, # (7p, 7p, 7p) => 7p
    18, # (8p, 8p, 8p) => 8p
    19, # (9p, 9p, 9p) => 9p
    21, # (1s, 1s, 1s) => 1s
    22, # (2s, 2s, 2s) => 2s
    23, # (3s, 3s, 3s) => 3s
    24, # (4s, 4s, 4s) => 4s
    20, # (5s, 5s, 5s) => 0s
    25, # (0s, 5s, 5s) => 5s
    25, # (5s, 5s, 0s) => 5s
    26, # (6s, 6s, 6s) => 6s
    27, # (7s, 7s, 7s) => 7s
    28, # (8s, 8s, 8s) => 8s
    29, # (9s, 9s, 9s) => 9s
    30, # (1z, 1z, 1z) => 1z
    31, # (2z, 2z, 2z) => 2z
    32, # (3z, 3z, 3z) => 3z
    33, # (4z, 4z, 4z) => 4z
    34, # (5z, 5z, 5z) => 5z
    35, # (6z, 6z, 6z) => 6z
    36  # (7z, 7z, 7z) => 7z
)


_NUM2DAMINGGANG = (
    ('5mr', ['5m', '5m', '5m']),
    ('1m',  ['1m', '1m', '1m']),
    ('2m',  ['2m', '2m', '2m']),
    ('3m',  ['3m', '3m', '3m']),
    ('4m',  ['4m', '4m', '4m']),
    ('5m',  ['5mr', '5m', '5m']),
    ('6m',  ['6m', '6m', '6m']),
    ('7m',  ['7m', '7m', '7m']),
    ('8m',  ['8m', '8m', '8m']),
    ('9m',  ['9m', '9m', '9m']),
    ('5pr', ['5p', '5p', '5p']),
    ('1p',  ['1p', '1p', '1p']),
    ('2p',  ['2p', '2p', '2p']),
    ('3p',  ['3p', '3p', '3p']),
    ('4p',  ['4p', '4p', '4p']),
    ('5p',  ['5pr', '5p', '5p']),
    ('6p',  ['6p', '6p', '6p']),
    ('7p',  ['7p', '7p', '7p']),
    ('8p',  ['8p', '8p', '8p']),
    ('9p',  ['9p', '9p', '9p']),
    ('5sr', ['5s', '5s', '5s']),
    ('1s',  ['1s', '1s', '1s']),
    ('2s',  ['2s', '2s', '2s']),
    ('3s',  ['3s', '3s', '3s']),
    ('4s',  ['4s', '4s', '4s']),
    ('5s',  ['5sr', '5s', '5s']),
    ('6s',  ['6s', '6s', '6s']),
    ('7s',  ['7s', '7s', '7s']),
    ('8s',  ['8s', '8s', '8s']),
    ('9s',  ['9s', '9s', '9s']),
    ('E',   ['E', 'E', 'E']),
    ('S',   ['S', 'S', 'S']),
    ('W',   ['W', 'W', 'W']),
    ('N',   ['N', 'N', 'N']),
    ('P',   ['P', 'P', 'P']),
    ('F',   ['F', 'F', 'F']),
    ('C',   ['C', 'C', 'C'])
)


_DAMINGGANG2NUM = {}
for i, (tile, consumed) in enumerate(_NUM2DAMINGGANG):
    k = (tile, tuple(consumed))
    _DAMINGGANG2NUM[k] = i


_DAMINGGANG_COUNTS = []
for tile, consumed in _NUM2DAMINGGANG:
    counts = {}
    for t in consumed:
        t = _TILE2NUM[t]
        if t not in counts:
            counts[t] = 0
        counts[t] += 1
    _DAMINGGANG_COUNTS.append(counts)


_NUM2ANGANG = (
    ['1m', '1m', '1m', '1m'],
    ['2m', '2m', '2m', '2m'],
    ['3m', '3m', '3m', '3m'],
    ['4m', '4m', '4m', '4m'],
    ['5mr', '5m', '5m', '5m'],
    ['6m', '6m', '6m', '6m'],
    ['7m', '7m', '7m', '7m'],
    ['8m', '8m', '8m', '8m'],
    ['9m', '9m', '9m', '9m'],
    ['1p', '1p', '1p', '1p'],
    ['2p', '2p', '2p', '2p'],
    ['3p', '3p', '3p', '3p'],
    ['4p', '4p', '4p', '4p'],
    ['5pr', '5p', '5p', '5p'],
    ['6p', '6p', '6p', '6p'],
    ['7p', '7p', '7p', '7p'],
    ['8p', '8p', '8p', '8p'],
    ['9p', '9p', '9p', '9p'],
    ['1s', '1s', '1s', '1s'],
    ['2s', '2s', '2s', '2s'],
    ['3s', '3s', '3s', '3s'],
    ['4s', '4s', '4s', '4s'],
    ['5sr', '5s', '5s', '5s'],
    ['6s', '6s', '6s', '6s'],
    ['7s', '7s', '7s', '7s'],
    ['8s', '8s', '8s', '8s'],
    ['9s', '9s', '9s', '9s'],
    ['E', 'E', 'E', 'E'],
    ['S', 'S', 'S', 'S'],
    ['W', 'W', 'W', 'W'],
    ['N', 'N', 'N', 'N'],
    ['P', 'P', 'P', 'P'],
    ['F', 'F', 'F', 'F'],
    ['C', 'C', 'C', 'C']
)


_ANGANG2NUM = {}
for i, consumed in enumerate(_NUM2ANGANG):
    k = tuple(consumed)
    _ANGANG2NUM[k] = i


_ANGANG_COUNTS = []
for consumed in _NUM2ANGANG:
    counts = {}
    for tile in consumed:
        tile = _TILE2NUM[tile]
        if tile not in counts:
            counts[tile] = 0
        counts[tile] += 1
    _ANGANG_COUNTS.append(counts)


_JIAGANG_LIST = (
     1, # (1m, 1m, 1m) + 1m
     2, # (2m, 2m, 2m) + 2m
     3, # (3m, 3m, 3m) + 3m
     4, # (4m, 4m, 4m) + 4m
     0, # (5m, 5m, 5m) + 0m
     5, # (0m, 5m, 5m) + 5m
     5, # (5m, 5m, 0m) + 5m
     6, # (6m, 6m, 6m) + 6m
     7, # (7m, 7m, 7m) + 7m
     8, # (8m, 8m, 8m) + 8m
     9, # (9m, 9m, 9m) + 9m
    11, # (1p, 1p, 1p) + 1p
    12, # (2p, 2p, 2p) + 2p
    13, # (3p, 3p, 3p) + 3p
    14, # (4p, 4p, 4p) + 4p
    10, # (5p, 5p, 5p) + 0p
    15, # (0p, 5p, 5p) + 5p
    15, # (5p, 5p, 0p) + 5p
    16, # (6p, 6p, 6p) + 6p
    17, # (7p, 7p, 7p) + 7p
    18, # (8p, 8p, 8p) + 8p
    19, # (9p, 9p, 9p) + 9p
    21, # (1s, 1s, 1s) + 1s
    22, # (2s, 2s, 2s) + 2s
    23, # (3s, 3s, 3s) + 3s
    24, # (4s, 4s, 4s) + 4s
    20, # (5s, 5s, 5s) + 0s
    25, # (0s, 5s, 5s) + 5s
    25, # (5s, 5s, 0s) + 5s
    26, # (6s, 6s, 6s) + 6s
    27, # (7s, 7s, 7s) + 7s
    28, # (8s, 8s, 8s) + 8s
    29, # (9s, 9s, 9s) + 9s
    30, # (1z, 1z, 1z) + 1z
    31, # (2z, 2z, 2z) + 2z
    32, # (3z, 3z, 3z) + 3z
    33, # (4z, 4z, 4z) + 4z
    34, # (5z, 5z, 5z) + 5z
    35, # (6z, 6z, 6z) + 6z
    36  # (7z, 7z, 7z) + 7z
)


_NUM2JIAGANG = (
    ('5mr', ['5m', '5m', '5m']),
    ('1m',  ['1m', '1m', '1m']),
    ('2m',  ['2m', '2m', '2m']),
    ('3m',  ['3m', '3m', '3m']),
    ('4m',  ['4m', '4m', '4m']),
    ('5m',  ['5mr', '5m', '5m']),
    ('6m',  ['6m', '6m', '6m']),
    ('7m',  ['7m', '7m', '7m']),
    ('8m',  ['8m', '8m', '8m']),
    ('9m',  ['9m', '9m', '9m']),
    ('5pr', ['5p', '5p', '5p']),
    ('1p',  ['1p', '1p', '1p']),
    ('2p',  ['2p', '2p', '2p']),
    ('3p',  ['3p', '3p', '3p']),
    ('4p',  ['4p', '4p', '4p']),
    ('5p',  ['5pr', '5p', '5p']),
    ('6p',  ['6p', '6p', '6p']),
    ('7p',  ['7p', '7p', '7p']),
    ('8p',  ['8p', '8p', '8p']),
    ('9p',  ['9p', '9p', '9p']),
    ('5sr', ['5s', '5s', '5s']),
    ('1s',  ['1s', '1s', '1s']),
    ('2s',  ['2s', '2s', '2s']),
    ('3s',  ['3s', '3s', '3s']),
    ('4s',  ['4s', '4s', '4s']),
    ('5s',  ['5sr', '5s', '5s']),
    ('6s',  ['6s', '6s', '6s']),
    ('7s',  ['7s', '7s', '7s']),
    ('8s',  ['8s', '8s', '8s']),
    ('9s',  ['9s', '9s', '9s']),
    ('E',   ['E', 'E', 'E']),
    ('S',   ['S', 'S', 'S']),
    ('W',   ['W', 'W', 'W']),
    ('N',   ['N', 'N', 'N']),
    ('P',   ['P', 'P', 'P']),
    ('F',   ['F', 'F', 'F']),
    ('C',   ['C', 'C', 'C'])
)


_JIAGANG_TO_PENG_LIST = (
    ( 4,),
    ( 0,),
    ( 1,),
    ( 2,),
    ( 3,),
    ( 5,  6,),
    ( 7,),
    ( 8,),
    ( 9,),
    (10,),
    (15,),
    (11,),
    (12,),
    (13,),
    (14,),
    (16, 17,),
    (18,),
    (19,),
    (20,),
    (21,),
    (26,),
    (22,),
    (23,),
    (24,),
    (25,),
    (27, 28,),
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
    (39,)
)


_TILE_TO_JIAGANG_COUNTS = []
for i, v in enumerate(_NUM2JIAGANG):
    tile, consumed = v
    counts = {}
    for t in consumed:
        if t not in counts:
            counts[t] = 0
        counts[t] += 1
    _TILE_TO_JIAGANG_COUNTS.append(counts)


class GameState:
    def __init__(
        self, *, my_name: str, room: int, game_style: int, my_grade: int,
        opponent_grade: int) -> None:
        self.__my_name = my_name
        self.__room = room
        self.__game_style = game_style
        self.__my_grade = my_grade
        self.__opponent_grade = opponent_grade
        self.__seat = None
        self.__player_grades = None
        self.__player_scores = None

    def on_new_game(self) -> None:
        pass

    def on_new_round(self, seat: int, scores: List[int]) -> None:
        self.__seat = seat

        self.__player_grades = [None] * 4
        for i in range(4):
            if i == self.__seat:
                self.__player_grades[i] = self.__my_grade
            else:
                self.__player_grades[i] = self.__opponent_grade

        self.__player_scores = list(scores)

    def __assert_initialized(self) -> None:
        if self.__player_grades is None:
            raise RuntimeError(
                'A method is called on a non-initialized `GameState` object.')
        assert(self.__player_scores is not None)

    def on_liqi_acceptance(self, seat: int) -> None:
        self.__assert_initialized()
        self.__player_scores[seat] -= 1000

    def get_my_name(self) -> str:
        self.__assert_initialized()
        return self.__my_name

    def get_room(self) -> int:
        self.__assert_initialized()
        return self.__room

    def get_game_style(self) -> int:
        self.__assert_initialized()
        return self.__game_style

    def get_seat(self) -> int:
        self.__assert_initialized()
        return self.__seat

    def get_player_grade(self, seat: int) -> int:
        self.__assert_initialized()
        return self.__player_grades[seat]

    def get_player_rank(self, seat: int) -> int:
        self.__assert_initialized()

        score = self.__player_scores[seat]
        rank = 0
        for i in range(seat):
            if self.__player_scores[i] >= score:
                rank += 1
        for i in range(seat + 1, 4):
            if self.__player_scores[i] > score:
                rank += 1
        assert(0 <= rank and rank < 4)
        return rank

    def get_player_score(self, seat: int) -> int:
        self.__assert_initialized()
        return self.__player_scores[seat]

class RoundState:
    def __init__(self) -> None:
        self.__xiangting_calculator = XiangtingCalculator('/workspace')
        self.__hand_calculator = HandCalculator()
        self.__chang = None
        self.__index = None
        self.__ben_chang = None
        self.__deposits = None
        self.__dora_indicators = None
        self.__num_left_tiles = None
        self.__my_hand = None
        self.__my_fulu_list = None
        self.__zimo_pai = None
        self.__my_first_zimo = None
        self.__liqi_to_be_accepted = None
        self.__my_liqi = None
        self.__my_lingshang_zimo = None
        self.__my_kuikae_tiles = None
        self.__my_zhenting = None
        self.__progression = None

    def on_new_round(
        self, chang: int, index: int, ben_chang: int, deposits: int,
        dora_indicator: int, hand: List[int]) -> None:
        self.__chang = chang
        self.__index = index
        self.__ben_chang = ben_chang
        self.__deposits = deposits
        self.__dora_indicators = [dora_indicator]
        self.__num_left_tiles = 70
        self.__my_hand = hand
        self.__my_fulu_list = []
        self.__zimo_pai = None
        self.__my_first_zimo = True
        self.__liqi_to_be_accepted = [False, False, False, False]
        self.__my_liqi = False
        self.__my_lingshang_zimo = False
        self.__my_kuikae_tiles = []
        # self.__my_zhenting == 1: 非立直中の栄和拒否による一時的なフリテン
        # self.__my_zhenting == 2: 立直中の栄和拒否による永続的なフリテン
        self.__my_zhenting = 0
        self.__progression = [0]

    def get_chang(self) -> int:
        return self.__chang

    def get_index(self) -> int:
        return self.__index

    def get_num_ben_chang(self) -> int:
        return self.__ben_chang

    def get_num_deposits(self) -> int:
        return self.__deposits

    def get_dora_indicators(self) -> List[int]:
        return self.__dora_indicators

    def get_num_left_tiles(self) -> int:
        return self.__num_left_tiles

    def get_my_hand(self) -> List[int]:
        return self.__my_hand

    def get_my_fulu_list(self) -> List[int]:
        return self.__my_fulu_list

    def get_zimo_tile(self) -> Optional[int]:
        return self.__zimo_pai

    def is_in_liqi(self) -> bool:
        return self.__my_liqi

    def copy_progression(self) -> List[int]:
        return list(self.__progression)

    def __get_my_hand_counts(self) -> Counter:
        my_hand_counts = Counter()
        for tile in self.__my_hand:
            my_hand_counts[tile] += 1
        return my_hand_counts

    def __set_my_hand_counts(self, hand_counts: Counter) -> None:
        self.__my_hand = []
        for k, v in hand_counts.items():
            for i in range(v):
                self.__my_hand.append(k)
        self.__my_hand.sort()
        if len(self.__my_hand) not in (1, 2, 4, 5, 7, 8, 10, 11, 13):
            raise RuntimeError('An invalid hand.')

    def on_zimo(
        self, seat: int, mine: bool, tile: Optional[int],
        my_score: int) -> Optional[List[int]]:
        if self.__zimo_pai is not None:
            raise AssertionError(f'self.__zimo_pai = {self.__zimo_pai}')

        self.__num_left_tiles -= 1
        self.__my_kuikae_tiles = []

        if not mine:
            if tile is not None:
                raise ValueError(f'tile = {tile}')
            return None

        if tile is None:
            raise ValueError(f'TODO: (A suitable error message)')
        self.__zimo_pai = tile

        # 非立直中の栄和拒否による一時的なフリテンを解消する．
        if self.__my_zhenting == 1:
            self.__my_zhenting = 0

        candidates = []

        if self.__my_liqi:
            # 立直中の場合．自摸切りを候補に追加する．
            candidates.append(self.__zimo_pai * 4 + 1 * 2 + 0)
        else:
            # 以下，立直中でない場合．
            # 手出しを候補として追加する．
            for i in range(len(self.__my_hand)):
                tile = self.__my_hand[i]
                candidates.append(tile * 4 + 0 * 2 + 0)
                new_hand = list(self.__my_hand)
                new_hand[i] = self.__zimo_pai
                if len(self.__my_fulu_list) == 0 and my_score >= 1000:
                    xiangting_number = self.__xiangting_calculator.calculate(new_hand, 4)
                    if xiangting_number == 1:
                        # 立直宣言を伴う手出しを候補として追加する．
                        candidates.append(tile * 4 + 0 * 2 + 1)

            # 自摸切りを候補として追加する．
            candidates.append(self.__zimo_pai * 4 + 1 * 2 + 0)
            if len(self.__my_fulu_list) == 0 and my_score >= 1000:
                xiangting_number = self.__xiangting_calculator.calculate(self.__my_hand, 4)
                if xiangting_number == 1:
                    # 立直宣言を伴う自摸切りを候補として追加する．
                    candidates.append(self.__zimo_pai * 4 + 1 * 2 + 1)

        combined_hand = self.__my_hand + [self.__zimo_pai]

        # 暗槓が候補として追加できるかどうかをチェックする．
        # TODO: 立直中の送り槓を禁止する．
        counts = Counter()
        for p in combined_hand:
            if p == 0:
                counts[4] += 1
            elif 1 <= p and p <= 9:
                counts[p - 1] += 1
            elif p == 10:
                counts[13] += 1
            elif 11 <= p and p <= 19:
                counts[p - 2] += 1
            elif p == 20:
                counts[22] += 1
            elif 21 <= p:
                assert(p < 37)
                counts[p - 3] += 1
        for k, v in counts.items():
            if v >= 4:
                candidates.append(148 + k)

        # 加槓が候補として追加できるかどうかをチェックする．
        peng_list = []
        for fulu in self.__my_fulu_list:
            if 312 <= fulu and fulu <= 431:
                peng = (fulu - 312) % 40
                peng_list.append(peng)
        for peng, t in enumerate(_JIAGANG_LIST):
            if peng in peng_list and t in combined_hand:
                candidates.append(182 + t)

        # 自摸和が候補として追加できるかどうかをチェックする．
        xiangting_number = self.__xiangting_calculator.calculate(
            combined_hand, 4 - len(self.__my_fulu_list))
        if xiangting_number == 0:
            player_wind = (seat + 4 - self.__index) % 4
            has_yihan = self.__hand_calculator.has_yihan(
                self.__chang, player_wind, self.__my_hand, self.__my_fulu_list,
                self.__zimo_pai, rong=False)
            if self.__my_liqi or self.__num_left_tiles == 0 or self.__my_lingshang_zimo or has_yihan:
                candidates.append(219)

        if self.__my_first_zimo:
            assert(not self.__my_liqi)
            # 九種九牌が候補として追加できるかどうかをチェックする．
            count = 0
            for p in set(combined_hand):
                if p in (1, 9, 11, 19, 21, 29, 30, 31, 32, 33, 34, 35, 36):
                    count += 1
            if count >= 9:
                candidates.append(220)

        self.__my_first_zimo = False
        self.__my_lingshang_zimo = False

        candidates = list(set(candidates))
        candidates.sort()
        return candidates

    def __get_my_zhenting_tiles(self, seat: int) -> List[int]:
        # 自分が捨てた牌を列挙する．
        discarded_tiles = set()
        for p in self.__progression:
            if p < 5 or 596 < p:
                continue
            encode = p - 5
            actor = encode // 148
            encode = encode % 148
            tile = encode // 4
            if actor != seat:
                continue
            discarded_tiles.add(tile)

        # 和牌の候補を列挙する．
        hupai_candidates = []
        for i in range(37):
            combined_hand = self.__my_hand + [i]
            xiangting_number = self.__xiangting_calculator.calculate(
                combined_hand, 4 - len(self.__my_fulu_list))
            if xiangting_number == 0:
                hupai_candidates.append(i)

        # 和牌の候補の中に自分が捨てた牌が1つでも含まれているならば，
        # 和牌の候補全てがフリテンの対象でありロンできない．
        for hupai_candidate in hupai_candidates:
            if hupai_candidate in discarded_tiles:
                return hupai_candidates
        return []

    def on_dapai(
        self, seat: int, actor: int, tile: int,
        moqi: bool) -> Optional[List[int]]:
        if self.__num_left_tiles == 69:
            # 雀魂から学習したモデルは親の第1打牌が必ず手出しになる．
            moqi = False

        liqi = self.__liqi_to_be_accepted[seat]

        encode = 5 + actor * 148 + tile * 4 + (2 if moqi else 0) + (1 if liqi else 0)
        self.__progression.append(encode)

        if actor == seat:
            if moqi:
                if self.__zimo_pai is None:
                    raise RuntimeError('TODO: (A suitable error message)')
                if self.__zimo_pai != tile:
                    raise RuntimeError('TODO: (A suitable error message)')
                self.__zimo_pai = None
                return None
            index = None
            for i in range(len(self.__my_hand)):
                if self.__my_hand[i] == tile:
                    index = i
                    break
            if index is None:
                # 自分が親の時の第1打牌で自摸切りの場合．
                if self.__num_left_tiles != 69:
                    raise RuntimeError('TODO: (A suitable error message)')
                if self.__zimo_pai is None:
                    raise RuntimeError('TODO: (A suitable error message)')
                if self.__zimo_pai != tile:
                    raise RuntimeError('TODO: (A suitable error message)')
                self.__zimo_pai = None
                return None
            self.__my_hand.pop(index)
            if self.__zimo_pai is not None:
                self.__my_hand.append(self.__zimo_pai)
                self.__zimo_pai = None
                self.__my_hand.sort()
            assert(len(self.__my_hand) in (1, 4, 7, 10, 13))
            return None

        relseat = (actor + 4 - seat) % 4 - 1

        skippable = False

        hand_counts = self.__get_my_hand_counts()

        candidates = []

        if not self.__my_liqi and relseat == 2:
            # チーができるかどうかチェックする．
            # TODO: 河底牌に対するチーが可能かどうか確認する．
            for i, (t, consumed_counts) in enumerate(_CHI_COUNTS):
                if tile != t:
                    continue
                new_hand_counts = Counter(hand_counts)
                for k, v in consumed_counts.items():
                    if hand_counts[k] < v:
                        new_hand_counts = None
                        break
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
                        self.__my_kuikae_tiles = list(_CHI_TO_KUIKAE_TILES[i])
                        candidates.append(222 + i)
                        skippable = True

        if not self.__my_liqi:
            # ポンができるかどうかチェックする．
            # TODO: 河底牌に対するポンが可能かどうか確認する．
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
                        self.__my_kuikae_tiles = [_PENG_TO_KUIKAE_TILE[i]]
                        candidates.append(312 + relseat * 40 + i)
                        skippable = True

        if not self.__my_liqi:
            # 大明槓ができるかどうかチェックする．
            # TODO: 河底牌に対する大明槓が可能かどうか確認する．
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

        combined_hand = self.__my_hand + [tile]

        xiangting_number = self.__xiangting_calculator.calculate(
            combined_hand, 4 - len(self.__my_fulu_list))
        if xiangting_number == 0 and tile not in self.__get_my_zhenting_tiles(seat) and self.__my_zhenting == 0:
            # ロンが出来るかどうかチェックする．
            player_wind = (seat + 4 - self.__index) % 4
            has_yihan = self.__hand_calculator.has_yihan(
                self.__chang, player_wind, self.__my_hand, self.__my_fulu_list,
                tile, rong=True)
            if self.__my_liqi or self.__num_left_tiles == 0 or has_yihan:
                candidates.append(543 + relseat)
                skippable = True

        if skippable:
            candidates.append(221)

        candidates.sort()
        return candidates if len(candidates) > 0 else None

    def on_chi(self, mine: bool, seat: int, chi: int) -> Optional[List[int]]:
        self.__my_first_zimo = False
        self.__progression.append(597 + seat * 90 + chi)

        if not mine:
            self.__my_kuikae_tiles = []
            return None

        my_hand_counts = self.__get_my_hand_counts()
        consumed_counts = _CHI_COUNTS[chi][1]
        for k, v in consumed_counts.items():
            if my_hand_counts[k] < v:
                raise RuntimeError('An invalid chi.')
            my_hand_counts[k] -= v
        self.__set_my_hand_counts(my_hand_counts)

        if len(self.__my_fulu_list) == 4:
            raise RuntimeError('An invalid chi.')
        self.__my_fulu_list.append(222 + chi)

        candidates = []
        for tile in self.__my_hand:
            if tile not in self.__my_kuikae_tiles:
                candidates.append(tile * 4 + 0 * 2 + 0)
        self.__my_kuikae_tiles = []
        return list(set(candidates))

    def on_peng(
        self, mine: bool, seat: int, relseat: int,
        peng: int) -> Optional[List[int]]:
        self.__my_first_zimo = False
        self.__progression.append(957 + seat * 120 + relseat * 40 + peng)

        if not mine:
            self.__my_kuikae_tiles = []
            return None

        my_hand_counts = self.__get_my_hand_counts()
        consumed_counts = _PENG_COUNTS[peng][1]
        for k, v in consumed_counts.items():
            if my_hand_counts[k] < v:
                raise RuntimeError('An invalid peng.')
            my_hand_counts[k] -= v
        self.__set_my_hand_counts(my_hand_counts)

        if len(self.__my_fulu_list) == 4:
            raise RuntimeError('An invalid peng.')
        self.__my_fulu_list.append(312 + relseat * 40 + peng)

        candidates = []
        for tile in self.__my_hand:
            if tile not in self.__my_kuikae_tiles:
                candidates.append(tile * 4 + 0 * 2 + 0)
        self.__my_kuikae_tiles = []
        return list(set(candidates))

    def on_daminggang(
        self, mine: bool, seat: int, relseat: int, daminggang: int) -> None:
        self.__my_first_zimo = False
        self.__my_kuikae_tiles = []
        self.__progression.append(1437 + seat * 111 + relseat * 37 + daminggang)

        if not mine:
            return

        my_hand_counts = self.__get_my_hand_counts()
        consumed_counts = _DAMINGGANG_COUNTS[daminggang]
        for k, v in consumed_counts.items():
            if my_hand_counts[k] < v:
                raise RuntimeError('An invalid daminggang.')
            my_hand_counts[k] -= v
        self.__set_my_hand_counts(my_hand_counts)

        if len(self.__my_fulu_list) == 4:
            raise RuntimeError('An invalid daminggang.')
        self.__my_fulu_list.append(432 + relseat * 37 + daminggang)

        self.__my_lingshang_zimo = True

    def on_angang(
        self, seat: int, actor: int, angang: int) -> Optional[List[int]]:
        self.__my_first_zimo = False
        self.__progression.append(1881 + actor * 34 + angang)

        if seat != actor:
            # TODO: 暗槓に対する国士無双の槍槓をチェックする．
            return None

        if self.__zimo_pai is None:
            raise RuntimeError('TODO: (A suitable error message)')

        my_hand_counts = self.__get_my_hand_counts()
        my_hand_counts[self.__zimo_pai] += 1
        consumed_counts = _ANGANG_COUNTS[angang]
        for k, v in consumed_counts.items():
            if my_hand_counts[k] < v:
                raise RuntimeError('An invalid angang.')
            my_hand_counts[k] -= v
        self.__set_my_hand_counts(my_hand_counts)
        self.__zimo_pai = None

        if len(self.__my_fulu_list) == 4:
            raise RuntimeError('An invalid angang.')
        self.__my_fulu_list.append(148 + angang)

        self.__my_lingshang_zimo = True

        return None

    def on_jiagang(
        self, seat: int, actor: int, tile: int) -> Optional[List[int]]:
        self.__my_first_zimo = False
        self.__progression.append(2017 + seat * 37 + tile)

        if seat != actor:
            # 槍槓が可能かどうかをチェックする．
            combined_hand = self.__my_hand + [tile]
            xiangting_number = self.__xiangting_calculator.calculate(
                combined_hand, 4 - len(self.__my_fulu_list))
            if xiangting_number == 0:
                relseat = (actor + 4 - seat) % 4 - 1
                return [221, 543 + relseat]
            return None

        if self.__zimo_pai is None:
            raise RuntimeError('TODO: A suitable error message')

        index = None
        for i in range(len(self.__my_hand)):
            if self.__my_hand[i] == tile:
                index = i
                break
        if index is not None:
            self.__my_hand.pop(index)
            self.__my_hand.append(self.__zimo_pai)
            self.__zimo_pai = None
            self.__my_hand.sort()
        else:
            if self.__zimo_pai != tile:
                raise RuntimeError('TODO: A suitable error message')
            self.__zimo_pai = None

        index = None
        for i in range(len(self.__my_fulu_list)):
            # 加槓の対象となるポンを探す．
            fulu = self.__my_fulu_list[i]
            if fulu < 312 or 431 < fulu:
                # ポンではない．
                continue
            peng = (fulu - 312) % 40
            if peng in _JIAGANG_TO_PENG_LIST[tile]:
                index = i
                break
        if index is None:
            raise RuntimeError('TODO: (A suitable error message)')
        self.__my_fulu_list[index] = (self.__my_fulu_list[index] - 312) % 40 + 182

        self.__my_lingshang_zimo = True

        return None

    def on_liqi(self, seat: int) -> None:
        if any(self.__liqi_to_be_accepted):
            raise RuntimeError('TODO: (A suitable error message)')
        self.__liqi_to_be_accepted[seat] = True

    def on_liqi_acceptance(self, mine: bool, seat: int) -> None:
        self.__deposits += 1

        if not self.__liqi_to_be_accepted[seat]:
            raise RuntimeError('TODO: (A suitable error message)')
        self.__liqi_to_be_accepted[seat] = False

        if mine:
            self.__my_liqi = True

    def on_new_dora(self, tile: int) -> None:
        if len(self.__dora_indicators) >= 5:
            raise RuntimeError(self.__dora_indicators)
        self.__dora_indicators.append(tile)

    def set_zhenting(self, zhenting: int) -> None:
        if zhenting not in (1, 2):
            raise ValueError('TODO: (A suitable error message)')
        self.__my_zhenting = zhenting


class Kanachan:
    def __init__(self) -> None:
        with open('./encoder.json', encoding='UTF-8') as f:
            encoder_config = json.load(f)
        encoder = Encoder(**encoder_config)
        encoder.cpu()
        encoder_snapshot_path = Path('./encoder.pth')
        encoder_state_dict = torch.load(
            encoder_snapshot_path, map_location=torch.device('cpu'))
        load_state_dict(encoder, encoder_state_dict)
        encoder.eval()

        with open('./decoder.json', encoding='UTF-8') as f:
            decoder_config = json.load(f)
        decoder = PolicyDecoder(**decoder_config)
        decoder.cpu()
        decoder_snapshot_path = Path('./decoder.pth')
        decoder_state_dict = torch.load(
            decoder_snapshot_path, map_location=torch.device('cpu'))
        load_state_dict(decoder, decoder_state_dict)
        decoder.eval()

        self.__model = PolicyModel(encoder, decoder)

        with open('./game.json', encoding='UTF-8') as f:
            game_config = json.load(f)
        self.__game_state = GameState(**game_config)

        self.__round_state = RoundState()

    def __on_hello(self) -> None:
        assert(message['type'] == 'hello')

        if 'can_act' not in message:
            raise RuntimeError('A `hello` message without the `can_act` key.')
        can_act = message['can_act']
        if not can_act:
            raise RuntimeError(
                f'A `hello` message with an invalid `can_act` (can_act = {can_act}).')

        my_name = self.get_my_name()
        response = json.dumps({'type': 'join', 'name': my_name, 'room': 'default'})
        print(response, flush=True)

    def __on_start_game(self, message: dict) -> None:
        assert(message['type'] == 'start_game')

        # 2022/10/06 時点でのAI雀荘の実装では， `start_game` メッセージに
        # `id` キーが伴わないように推定されるので，この時点では `seat` を
        # 設定できない．
        #
        #if 'id' not in message:
        #    raise RuntimeError('A `start_game` message without the `id` key.')
        #seat = message['id']
        #if seat < 0 or 4 <= seat:
        #    raise RuntimeError(
        #        f'A `start_game` message with an invalid `id` (id = {seat}).')
        #self.__game_state.on_new_game(seat)

        self.__game_state.on_new_game()

        response = json.dumps({'type': 'none'})
        print(response, flush=True)

    def __on_start_kyoku(self, message: dict) -> None:
        assert(message['type'] == 'start_kyoku')

        if 'bakaze' not in message:
            raise RuntimeError(
                'A `start_kyoku` message without the `bakaze` key.')
        chang = message['bakaze']
        if chang not in ('E', 'S', 'W'):
            raise RuntimeError(
                f'A `start_kyoku` message with an invalid `bakaze` (bakaze = {chang}).')
        chang = {'E': 0, 'S': 1, 'W': 2}[chang]

        if 'kyoku' not in message:
            raise RuntimeError(
                'A `start_kyoku` message without the `kyoku` key.')
        round_index = message['kyoku']
        if round_index < 1 or 4 < round_index:
            raise RuntimeError(
                f'A `start_kyoku` message with an invalid `kyoku` (kyoku = {round_index}).')
        round_index -= 1

        if 'honba' not in message:
            raise RuntimeError(
                'A `start_kyoku` message without the `honba` key.')
        ben_chang = message['honba']
        if ben_chang < 0:
            raise RuntimeError(
                f'A `start_kyoku` message with an invalid `honba` (honba = {ben_chang}).')

        if 'kyotaku' not in message:
            raise RuntimeError(
                'A `start_kyoku` message without the `kyotaku` key.')
        deposits = message['kyotaku']
        if deposits < 0:
            raise RuntimeError(
                f'A `start_kyoku` message with an invalid `kyotaku` (kyotaku = {deposits}).')

        if 'oya' not in message:
            raise RuntimeError('A `start_kyoku` message without the `oya` key.')
        dealer = message['oya']
        if dealer != round_index:
            raise RuntimeError(
                f'An inconsistent `start_kyoku` message (round_index = {round_index}, oya = {dealer}).')

        if 'dora_marker' not in message:
            raise RuntimeError(
                'A `start_kyoku` message without the `dora_marker` key.')
        dora_indicator = message['dora_marker']
        if dora_indicator not in _TILE2NUM:
            raise RuntimeError(
                'A `start_kyoku` message with an invalid `dora_marker` (dora_marker = {dora_indicator}).')
        dora_indicator = _TILE2NUM[dora_indicator]

        if 'scores' not in message:
            raise RuntimeError(
                'A `start_kyoku` message without the `scores` key.')
        scores = message['scores']
        if len(scores) != 4:
            raise RuntimeError(
                f'A `start_kyoku` message with an invalid scores (length = {len(scores)}).')

        if 'tehais' not in message:
            raise RuntimeError(
                'A `start_kyoku` message without the `tehais` key.')
        hands = message['tehais']
        if len(hands) != 4:
            raise RuntimeError(
                f'A `start_kyoku` message with an wrong `tehais` (length = {len(hands)}).')

        # 2022/10/06 時点でのAI雀荘の実装では， `start_game` メッセージに
        # `id` キーが伴わないように推定されるので，その時点では `seat` を
        # 設定できない．以下は workaround
        seat = None
        for i in range(4):
            if hands[i][0] != '?':
                if seat is not None:
                    raise RuntimeError('TODO: (A suitable error message)')
                seat = i
        if seat is None:
            raise RuntimeError('TODO: (A suitable error message)')

        for i, hand in enumerate(hands):
            if i != seat:
                if hand != ['?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?']:
                    raise RuntimeError(
                        f'A `start_kyoku` message with an wrong `tehais` (seat = {seat}, i = {i}, hand = {hand}).')
            else:
                if len(hand) != 13:
                    raise RuntimeError(
                        f'A `start_kyoku` message with an wrong `tehais` (seat = {seat}, hand = {hand}).')
                for i in range(13):
                    if hand[i] not in _TILE2NUM:
                        raise RuntimeError(
                            f'A `start_kyoku` message with an wrong `tehais` (seat = {seat}, hand = {hand}).')
                    hand[i] = _TILE2NUM[hand[i]]
        hand = hands[seat]

        self.__game_state.on_new_round(seat, scores)
        self.__round_state.on_new_round(chang, round_index, ben_chang, deposits, dora_indicator, hand)

    def __respond(self, dapai: Optional[int], candidates: List[int]) -> None:
        seat = self.__game_state.get_seat()

        sparse = []
        sparse.append(self.__game_state.get_room())
        sparse.append(self.__game_state.get_game_style() + 5)
        sparse.append(seat + 7)
        sparse.append(self.__round_state.get_chang() + 11)
        sparse.append(self.__round_state.get_index() + 14)
        for i, dora_indicator in enumerate(self.__round_state.get_dora_indicators()):
            sparse.append(dora_indicator + 37 * i + 18)
        sparse.append(self.__round_state.get_num_left_tiles() + 203)
        sparse.append(self.__game_state.get_player_grade(seat) + 273)
        sparse.append(self.__game_state.get_player_rank(seat) + 289)
        sparse.append(self.__game_state.get_player_grade((seat + 1) % 4) + 293)
        sparse.append(self.__game_state.get_player_rank((seat + 1) % 4) + 309)
        sparse.append(self.__game_state.get_player_grade((seat + 2) % 4) + 313)
        sparse.append(self.__game_state.get_player_rank((seat + 2) % 4) + 329)
        sparse.append(self.__game_state.get_player_grade((seat + 3) % 4) + 333)
        sparse.append(self.__game_state.get_player_rank((seat + 3) % 4) + 349)
        hand_encode = [None] * 136
        for tile in self.__round_state.get_my_hand():
            flag = False
            for i in range(_TILE_OFFSETS[tile], _TILE_OFFSETS[tile + 1]):
                if hand_encode[i] is None:
                    hand_encode[i] = 1
                    flag = True
                    break
            if not flag:
                raise RuntimeError('TODO: (A suitable error message)')
        for i in range(136):
            if hand_encode[i] == 1:
                sparse.append(i + 353)
        zimo_tile = self.__round_state.get_zimo_tile()
        if zimo_tile is not None:
            sparse.append(zimo_tile + 489)
        for i in range(len(sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
            sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
        sparse = torch.tensor(sparse, device='cpu', dtype=torch.int32)
        sparse = torch.unsqueeze(sparse, dim=0)

        numeric = []
        numeric.append(self.__round_state.get_num_ben_chang())
        numeric.append(self.__round_state.get_num_deposits())
        numeric.append(self.__game_state.get_player_score(seat) / 10000.0)
        numeric.append(self.__game_state.get_player_score((seat + 1) % 4) / 10000.0)
        numeric.append(self.__game_state.get_player_score((seat + 2) % 4) / 10000.0)
        numeric.append(self.__game_state.get_player_score((seat + 3) % 4) / 10000.0)
        numeric = torch.tensor(numeric, device='cpu', dtype=torch.float32)
        numeric = torch.unsqueeze(numeric, dim=0)

        progression = self.__round_state.copy_progression()
        for i in range(len(progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
            progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
        progression = torch.tensor(progression, device='cpu', dtype=torch.int32)
        progression = torch.unsqueeze(progression, dim=0)

        candidates_ = list(candidates)
        for i in range(len(candidates_), MAX_NUM_ACTION_CANDIDATES):
            candidates_.append(NUM_TYPES_OF_ACTIONS + 1)
        candidates_ = torch.tensor(candidates_, device='cpu', dtype=torch.int32)
        candidates_ = torch.unsqueeze(candidates_, dim=0)

        with torch.no_grad():
            prediction = self.__model((sparse, numeric, progression, candidates_))
            prediction = torch.squeeze(prediction, dim=0)
            prediction = prediction[:len(candidates)]
            argmax = torch.argmax(prediction)
            argmax = argmax.item()
        candidates_ = torch.squeeze(candidates_, dim=0)
        decision = candidates_[argmax].item()

        if 0 <= decision and decision <= 147:
            tile = decision // 4
            tile = _NUM2TILE[tile]
            encode = decision % 4
            moqi = encode // 2 == 1
            encode = encode % 2
            liqi = encode == 1

            if liqi:
                response = json.dumps({'type': 'reach', 'actor': seat})
                print(response, flush=True)
                messages = sys.stdin.readline()
                messages = json.loads(messages)
                if len(messages) > 1:
                    raise RuntimeError(
                        'Too many messages starting with `reach`.')
                message = messages[0]
                if 'type' not in message:
                    raise RuntimeError('A message without the `type` key.')
                message_type = message['type']
                if message_type != 'reach':
                    raise RuntimeError(
                        f'A `reach` message is expected, but got a `{message_type}` message')
                if 'actor' not in message:
                    raise RuntimeError(
                        'A `reach` message without the `actor` key.')
                actor = message['actor']
                if actor != seat:
                    raise RuntimeError(
                        f'A `reach` message with an invalid actor (actor = {actor}).')
                self.__round_state.on_liqi(seat)

            response = json.dumps({'type': 'dahai', 'actor': seat, 'pai': tile, 'tsumogiri': moqi})
            print(response, flush=True)
            return

        if 148 <= decision and decision <= 181:
            angang = _NUM2ANGANG[decision - 148]
            response = json.dumps({'type': 'ankan', 'actor': seat, 'consumed': angang})
            print(response, flush=True)
            return

        if 182 <= decision and decision <= 218:
            tile, consumed = _NUM2JIAGANG[decision - 182]
            response = json.dumps({'type': 'kakan', 'actor': seat, 'pai': tile, 'consumed': consumed})
            print(response, flush=True)
            return

        if decision == 219:
            hupai = self.__round_state.get_zimo_tile()
            if hupai is None:
                raise RuntimeError('Trying zimohu without any zimo tile.')
            hupai = _NUM2TILE[hupai]
            response = json.dumps({'type': 'hora', 'actor': seat, 'target': seat, 'pai': hupai})
            print(response, flush=True)
            return

        if decision == 220:
            response = json.dumps({'type': 'ryukyoku'})
            print(response, flush=True)
            return

        if decision == 221:
            response = json.dumps({'type': 'none'})
            print(response, flush=True)
            in_liqi = self.__round_state.is_in_liqi()
            for i in (543, 544, 545):
                if i in candidates:
                    # 栄和が選択肢にあるにも関わらず見逃しを選択した．
                    # この結果，フリテンが発生する．
                    self.__round_state.set_zhenting(2 if in_liqi else 1)
                    break
            return

        if 222 <= decision and decision <= 311:
            tile, consumed = _NUM2CHI[decision - 222]
            response = json.dumps({'type': 'chi', 'actor': seat, 'target': (seat + 3) % 4, 'pai': tile, 'consumed': consumed})
            print(response, flush=True)
            return

        if 312 <= decision and decision <= 431:
            encode = decision - 312
            relseat = encode // 40
            target = (seat + relseat + 1) % 4
            encode = encode % 40
            tile, consumed = _NUM2PENG[encode]
            response = json.dumps({'type': 'pon', 'actor': seat, 'target': target, 'pai': tile, 'consumed': consumed})
            print(response, flush=True)
            return

        if 432 <= decision and decision <= 542:
            encode = decision - 432
            relseat = encode // 37
            target = (seat + relseat + 1) % 4
            encode = encode % 37
            tile, consumed = _NUM2DAMINGGANG[encode]
            response = json.dumps({'type': 'daiminkan', 'actor': seat, 'target': target, 'pai': tile, 'consumed': consumed})
            print(response, flush=True)
            return

        if 543 <= decision and decision <= 545:
            relseat = decision - 543
            target = (seat + relseat + 1) % 4
            hupai = dapai
            if hupai is None:
                raise RuntimeError('Trying rong without any dapai.')
            hupai = _NUM2TILE[hupai]
            response = json.dumps({'type': 'hora', 'actor': seat, 'target': target, 'pai': hupai})
            print(response, flush=True)
            return

        raise RuntimeError(f'An invalid decision (decision = {decision}).')

    def __on_zimo(self, message: dict) -> None:
        assert(message['type'] == 'tsumo')

        seat = self.__game_state.get_seat()

        if 'actor' not in message:
            raise RuntimeError('A `tsumo` message without the `actor` key.')
        actor = message['actor']
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f'A `tsumo` message with an invalid `actor` (actor = {actor}).')
        mine = actor == seat

        if 'pai' not in message:
            raise RuntimeError('A `tsumo` message without the `pai` key.')
        tile = message['pai']

        my_score = self.__game_state.get_player_score(seat)

        if not mine:
            # TODO: https://github.com/smly/mjai.app/discussions/11
            #if tile != '?':
            #    raise RuntimeError(
            #        f'An inconsistent `tsumo` message (seat = {seat}, actor = {actor}, pai = {tile}).')
            self.__round_state.on_zimo(seat, mine, None, my_score)
        else:
            if tile not in _TILE2NUM:
                raise RuntimeError(
                    f'A `tsumo` message with an invalid `pai` (seat = {seat}, actor = {actor}, pai = {tile}).')
            tile = _TILE2NUM[tile]
            candidates = self.__round_state.on_zimo(seat, mine, tile, my_score)
            if not isinstance(candidates, list):
                raise RuntimeError(candidates)
            if len(candidates) == 0:
                raise RuntimeError('The length of `candidates` is equal to 0.')
            self.__respond(None, candidates)

    def __on_dapai(self, message: dict) -> None:
        assert(message['type'] == 'dahai')

        seat = self.__game_state.get_seat()

        if 'actor' not in message:
            raise RuntimeError('A `dahai` message without the `actor` key.')
        actor = message['actor']
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f'A `dahai` message with an invalid `actor` (actor = {actor}).')

        if 'pai' not in message:
            raise RuntimeError('A `dahai` message without the `pai` key.')
        tile = message['pai']
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f'A `dahai` message with an invalid `pai` (pai = {tile}).')
        tile = _TILE2NUM[tile]

        if 'tsumogiri' not in message:
            raise RuntimeError('A `dahai` message without the `tsumogiri` key.')
        moqi = message['tsumogiri']

        candidates = self.__round_state.on_dapai(seat, actor, tile, moqi)

        if actor == seat:
            # 自身の打牌に対してやることは何もない．
            if candidates is not None:
                raise RuntimeError(candidates)
            return

        if candidates is None:
            return

        if not isinstance(candidates, list):
            raise RuntimeError(candidates)
        if len(candidates) < 2:
            raise RuntimeError(candidates)
        self.__respond(tile, candidates)

    def __on_chi(self, message: dict) -> None:
        assert(message['type'] == 'chi')

        if 'actor' not in message:
            raise RuntimeError('A `chi` message without the `actor` key.')
        actor = message['actor']
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f'A `chi` message with an invalid `actor` (actor = {actor}).')
        mine = actor == self.__game_state.get_seat()

        if 'target' not in message:
            raise RuntimeError('A `chi` message without the `target` key.')
        target = message['target']
        if target < 0 or 4 <= target:
            raise RuntimeError(
                f'A `chi` message with an invalid `target` (target = {target}).')
        if (target + 4 - actor) % 4 != 3:
            raise RuntimeError(
                f'An inconsistent `chi` message (actor = {actor}, target = {target}).')

        if 'pai' not in message:
            raise RuntimeError('A `chi` message without the `pai` key.')
        tile = message['pai']
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f'A `chi` message with an invalid `pai` (pai = {tile}).')

        if 'consumed' not in message:
            raise RuntimeError('A `pon` message without the `consumed` key.')
        consumed = message['consumed']
        for t in consumed:
            if t not in _TILE2NUM:
                raise RuntimeError(
                    f'A `chi` message with an invalid `consumed` (consumed = {consumed}).')

        chi = (tile, tuple(consumed))
        if chi not in _CHI2NUM:
            raise RuntimeError(chi)
        chi = _CHI2NUM[chi]

        candidates = self.__round_state.on_chi(mine, actor, chi)

        if candidates is None:
            if mine:
                raise RuntimeError('TODO: (A suitable error message)')
            return

        if not isinstance(candidates, list):
            raise RuntimeError(candidates)
        if len(candidates) == 0:
            raise RuntimeError('TODO: (A suitable error message)')
        if not mine:
            raise RuntimeError('TODO: (A suitable error message)')
        self.__respond(None, candidates)

    def __on_peng(self, message: dict) -> None:
        assert(message['type'] == 'pon')

        if 'actor' not in message:
            raise RuntimeError('A `pon` message without the `actor` key.')
        actor = message['actor']
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f'A `pon` message with an invalid `actor` (actor = {actor}).')
        mine = actor == self.__game_state.get_seat()

        if 'target' not in message:
            raise RuntimeError('A `pon` message without the `target` key.')
        target = message['target']
        if target < 0 or 4 <= target:
            raise RuntimeError(
                f'A `pon` message with an invalid `target` (target = {target}).')
        if actor == target:
            raise RuntimeError(
                f'An inconsistent `pon` message (actor = {actor}, target = {target}).')
        relseat = (target + 4 - actor) % 4 - 1

        if 'pai' not in message:
            raise RuntimeError('A `pon` message without the `pai` key.')
        tile = message['pai']
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f'A `pon` message with an invalid `pai` (pai = {tile}).')

        if 'consumed' not in message:
            raise RuntimeError('A `pon` message without the `consumed` key.')
        consumed = message['consumed']
        for t in consumed:
            if t not in _TILE2NUM:
                raise RuntimeError(
                    f'A `pon` message with an invalid `consumed` (consumed = {consumed}).')

        peng = (tile, tuple(consumed))
        if peng not in _PENG2NUM:
            raise RuntimeError(peng)
        peng = _PENG2NUM[peng]

        candidates = self.__round_state.on_peng(mine, actor, relseat, peng)

        if candidates is None:
            if mine:
                raise RuntimeError('TODO: (A suitable error message)')
            return

        if not isinstance(candidates, list):
            raise RuntimeError(candidates)
        if len(candidates) == 0:
            raise RuntimeError('TODO: (A suitable error message)')
        if not mine:
            raise RuntimeError('TODO: (A suitable error message)')
        self.__respond(None, candidates)

    def __on_daminggang(self, message: dict) -> None:
        assert(message['type'] == 'daiminkan')

        if 'actor' not in message:
            raise RuntimeError('A `daiminkan` message without the `actor` key.')
        actor = message['actor']
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f'A `daiminkan` message with an invalid `actor` (actor = {actor}).')
        mine = actor == self.__game_state.get_seat()

        if 'target' not in message:
            raise RuntimeError(
                'A `daiminkan` message without the `target` key.')
        target = message['target']
        if target < 0 or 4 <= target:
            raise RuntimeError(
                f'A `daiminkan` message with an invalid `target` (target = {target}).')
        if actor == target:
            raise RuntimeError(
                f'An inconsistent `daiminkan` message (actor = {actor}, target = {target}).')
        relseat = (target + 4 - actor) % 4 - 1

        if 'pai' not in message:
            raise RuntimeError('A `daiminkan` message without the `pai` key.')
        tile = message['pai']
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f'A `daiminkan` message with an invalid `pai` (pai = {tile}).')

        if 'consumed' not in message:
            raise RuntimeError(
                'A `daiminkan` message without the `consumed` key.')
        consumed = message['consumed']
        for t in consumed:
            if t not in _TILE2NUM:
                raise RuntimeError(
                    f'A `daiminkan` message with an invalid `consumed` (consumed = {consumed}).')

        daminggang = (tile, tuple(consumed))
        if daminggang not in _DAMINGGANG2NUM:
            raise RuntimeError(daminggang)
        daminggang = _DAMINGGANG2NUM[daminggang]

        self.__round_state.on_daminggang(mine, actor, relseat, daminggang)

    def __on_angang(self, message: dict) -> None:
        assert(message['type'] == 'ankan')

        seat = self.__game_state.get_seat()

        if 'actor' not in message:
            raise RuntimeError('A `ankan` message without the `actor` key.')
        actor = message['actor']
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f'A `ankan` message with an invalid `actor` (actor = {actor}).')
        mine = actor == seat

        if 'consumed' not in message:
            raise RuntimeError(
                'A `ankan` message without the `consumed` key.')
        consumed = message['consumed']
        angang = tuple(consumed)

        if angang not in _ANGANG2NUM:
            raise RuntimeError(angang)
        angang = _ANGANG2NUM[angang]

        candidates = self.__round_state.on_angang(seat, actor, angang)
        if mine:
            if candidates is not None:
                raise RuntimeError(candidates)
            return

        if candidates is None:
            return

        if not isinstance(candidates, list):
            raise RuntimeError(candidates)
        if len(candidates) != 2:
            raise RuntimeError(len(candidates))
        # 国士無双の槍槓の可能性があるため，暗槓は打牌とみなす．
        dapai = _TILE2NUM[consumed[0]]
        self.__respond(dapai, candidates)

    def __on_jiagang(self, message: dict) -> None:
        assert(message['type'] == 'kakan')

        seat = self.__game_state.get_seat()

        if 'actor' not in message:
            raise RuntimeError('A `kakan` message without the `actor` key.')
        actor = message['actor']
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f'A `kakan` message with an invalid `actor` (actor = {actor}).')
        mine = actor == seat

        if 'pai' not in message:
            raise RuntimeError('A `kakan` message without the `pai` key.')
        tile = message['pai']
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f'A `kakan` message with an invalid `pai` (pai = {tile}).')
        tile = _TILE2NUM[tile]

        if 'consumed' not in message:
            raise RuntimeError('A `kakan` message without the `consumed` key.')
        consumed = message['consumed']

        candidates = self.__round_state.on_jiagang(seat, actor, tile)
        if mine:
            if candidates is not None:
                raise RuntimeError(candidates)
            return

        if candidates is None:
            return

        if not isinstance(candidates, list):
            raise RuntimeError(candidates)
        if len(candidates) != 2:
            raise RuntimeError(len(candidates))
        self.__respond(tile, candidates)

    def __on_liqi(self, message: dict) -> None:
        assert(message['type'] == 'reach')

        if 'actor' not in message:
            raise RuntimeError('A `reach` message without the `actor` key.')
        actor = message['actor']
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f'A `reach` message with an invalid `actor` (actor = {actor}).')

        self.__round_state.on_liqi(actor)

    def __on_liqi_acceptance(self, message: dict) -> None:
        assert(message['type'] == 'reach_accepted')

        if 'actor' not in message:
            raise RuntimeError(
                'A `reach_accepted` message without the `actor` key.')
        actor = message['actor']
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f'A `reach_accepted` message with an invalid `actor` (actor = {actor}).')
        mine = actor == self.__game_state.get_seat()

        self.__game_state.on_liqi_acceptance(actor)
        self.__round_state.on_liqi_acceptance(mine, actor)

    def __on_new_dora(self, message: dict) -> None:
        assert(message['type'] == 'dora')

        if 'dora_marker' not in message:
            raise RuntimeError(
                'A `dora` message without the `dora_marker` key.')
        dora_indicator = message['dora_marker']
        if dora_indicator not in _TILE2NUM:
            raise RuntimeError(
                f'A `dora` message with an invalid `dora_marker` (dora_marker = {dora_marker}).')
        dora_indicator = _TILE2NUM[dora_indicator]

        self.__round_state.on_new_dora(dora_indicator)

    def __on_hulu(self, message: dict) -> None:
        assert(message['type'] == 'hora')

        if 'actor' not in message:
            raise RuntimeError('A `hora` message without the `actor` key.')
        actor = message['actor']
        if actor < 0 or 4 <= actor:
            raise RuntimeError(
                f'A `hora` message with an invalid `actor` (actor = {actor}).')

        if 'target' not in message:
            raise RuntimeError('A `hora` message without the `target` key.')
        target = message['target']
        if target < 0 or 4 <= target:
            raise RuntimeError(
                f'A `hora` message with an invalid `target` (target = {target}).')

        if 'pai' not in message:
            raise RuntimeError('A `hora` message without the `pai` key.')
        tile = message['pai']
        if tile not in _TILE2NUM:
            raise RuntimeError(
                f'A `hora` message with an invalid `pai` (pai = {tile}).')

    def __on_luju(self, message: dict) -> None:
        assert(message['type'] == 'ryukyoku')

        if 'can_act' not in message:
            raise RuntimeError(
                'A `ryukyoku` message without the `can_act` key.')
        can_act = message['can_act']
        if can_act:
            raise RuntimeError(
                f'An inconsistent `ryukyoku` message (can_act = {can_act}).')

    def __on_round_end(self, message: dict) -> None:
        assert(message['type'] == 'end_kyoku')

        response = json.dumps({'type': 'none'})
        print(response, flush=True)

    def __on_game_end(self, message: dict) -> None:
        assert(message['type'] == 'end_game')

        response = json.dumps({'type': 'none'})
        print(response, flush=True)

    def run(self) -> None:
        messages = []
        while True:
            if len(messages) == 0:
                message_line = sys.stdin.readline()
                if message_line.strip() == '':
                    # workaround
                    continue
                messages = json.loads(message_line)
                if len(messages) == 0:
                    raise RuntimeError('The standard input is empty.')

            message = messages[0]
            if 'type' not in message:
                raise RuntimeError('A message without the `type` key.')

            if message['type'] == 'hello':
                if len(messages) > 1:
                    raise RuntimeError('A multi-line `hello` message.')
                self.__on_hello()
                messages.pop(0)
                continue

            if message['type'] == 'start_game':
                if len(messages) != 1:
                    raise RuntimeError(
                        'Too many messages starting with `start_game`.')
                self.__on_start_game(message)
                messages.pop(0)
                continue

            if message['type'] == 'start_kyoku':
                if len(messages) < 2:
                    raise RuntimeError(
                        'Too few messages starting with `start_kyoku`.')
                self.__on_start_kyoku(message)
                messages.pop(0)
                continue

            if message['type'] == 'tsumo':
                self.__on_zimo(message)
                messages.pop(0)
                continue

            if message['type'] == 'dahai':
                self.__on_dapai(message)
                messages.pop(0)
                continue

            if message['type'] == 'chi':
                self.__on_chi(message)
                messages.pop(0)
                continue

            if message['type'] == 'pon':
                self.__on_peng(message)
                messages.pop(0)
                continue

            if message['type'] == 'daiminkan':
                self.__on_daminggang(message)
                messages.pop(0)
                continue

            if message['type'] == 'ankan':
                self.__on_angang(message)
                messages.pop(0)
                continue

            if message['type'] == 'kakan':
                self.__on_jiagang(message)
                messages.pop(0)
                continue

            if message['type'] == 'reach':
                self.__on_liqi(message)
                messages.pop(0)
                continue

            if message['type'] == 'reach_accepted':
                self.__on_liqi_acceptance(message)
                messages.pop(0)
                continue

            if message['type'] == 'dora':
                self.__on_new_dora(message)
                messages.pop(0)
                continue

            if message['type'] == 'hora':
                self.__on_hulu(message)
                messages.pop(0)
                continue

            if message['type'] == 'ryukyoku':
                self.__on_luju(message)
                messages.pop(0)
                continue

            if message['type'] == 'end_kyoku':
                self.__on_round_end(message)
                messages.pop(0)
                continue

            if message['type'] == 'end_game':
                self.__on_game_end(message)
                messages.pop(0)
                if len(messages) > 0:
                    raise RuntimeError('TODO: (A suitable error message)')
                continue

            raise RuntimeError(message)
