#!/usr/bin/env python3

import re
from pathlib import Path
from argparse import ArgumentParser
import yaml
import sys
import importlib
import jsonschema
import torch
from torch import backends
from kanachan.training.bert.model_loader import load_model
from kanachan.training.bert.model_mode import ModelMode
from kanachan.simulation import simulate


def main():
    ap = ArgumentParser(description='Run Mahjong Soul game simulation.')
    ap.add_argument('--device', help='device', metavar='DEV')
    ap.add_argument(
        '--dtype', default='float16', choices=('float16', 'float32',),
        help='data type (defaults to `float16`)', metavar='DTYPE')
    ap.add_argument(
        '--baseline-model', type=Path, required=True,
        help='path to config file for baseline model', metavar='BASELINE_MODEL')
    ap.add_argument(
        '--baseline-grade', type=int, required=True,
        help='grade of baseline model', metavar='BASELINE_GRADE')
    ap.add_argument(
        '--proposed-model', type=Path, required=True,
        help='path to config file for proposed model', metavar='PROPOSED_MODEL')
    ap.add_argument(
        '--proposed-grade', type=int, required=True,
        help='grade of proposed model', metavar='PROPOSED_GRADE')
    ap.add_argument(
        '--room', choices=('bronze', 'silver', 'gold', 'jade', 'throne',),
        required=True, help='room for simulation')
    ap.add_argument(
        '--dongfengzhan', action='store_true',
        help='simulate dong feng zhan (東風戦)')
    ap.add_argument(
        '--mode', default='2vs2', choices=('2vs2', '1vs3',),
        help='simulation mode (defaults to `2vs2`)')
    ap.add_argument(
        '--non-duplicated', action='store_true',
        help='disable duplicated simulation')
    ap.add_argument(
        '-n', default=1, type=int,
        help='# of sets to simulate (defaults to `1`)',
        metavar='N')

    config = ap.parse_args()

    if config.device is not None:
        m = re.search('^(?:cpu|cuda(\\d*))$', config.device)
        if m is None:
            raise RuntimeError(f'{config.device}: invalid device')
        device = config.device
    elif backends.cuda.is_built():
        device = 'cuda'
    else:
        device = 'cpu'

    if config.dtype == 'float16':
        dtype = torch.float16
    elif config.dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError(f'{config.dtype}: An invalid data type.')

    if not config.baseline_model.exists():
        raise RuntimeError(f'{config.baseline_model}: Does not exist.')
    if not config.baseline_model.is_file():
        raise RuntimeError(f'{config.baseline_model}: Not a file.')
    baseline_model = load_model(config.baseline_model)
    baseline_model.to(device=device, dtype=dtype)

    if config.baseline_grade < 0 or 15 < config.baseline_grade:
        raise RuntimeError(
            f'{config.baseline_grade}: An invalid value for `--baseline-grade`.')

    if not config.proposed_model.exists():
        raise RuntimeError(f'{config.proposed_model}: Does not exist.')
    if not config.proposed_model.is_file():
        raise RuntimeError(f'{config.proposed_model}: Not a file.')
    proposed_model = load_model(config.proposed_model)
    proposed_model.to(device=device, dtype=dtype)

    if config.proposed_grade < 0 or 15 < config.proposed_grade:
        raise RuntimeError(
            f'{config.proposed_grade}: An invalid value for `--proposed-grade`.')

    room = {'bronze': 0, 'silver': 1, 'gold': 2, 'jade': 3, 'throne': 4}[config.room]

    mode = 0
    if config.non_duplicated:
        mode |= 1
    if config.dongfengzhan:
        mode |= 2
    if config.mode == '1vs3':
        mode |= 4
    mode |= (1 << (room + 3))

    if config.n < 1:
        raise RuntimeError(f'{config.n}: An invalid value for the `-n` option.')

    result = {
        'baseline': {
            'rounds': [],
            'games': []
        },
        'proposed': {
            'rounds': [],
            'games': []
        }
    }
    with ModelMode(baseline_model, 'prediction'), ModelMode(proposed_model, 'prediction'):
        for i in range(config.n):
            simulate(
                device, dtype, mode, config.baseline_grade, baseline_model,
                config.proposed_grade, proposed_model, result)

    num_games = 0
    ranking = 0.0
    grading_point = 0.0
    jade_point = 0.0
    num_tops = 0.0
    for game in result['proposed']['games']:
        r = game['ranking']
        s = game['score']
        ranking += r
        grading_point += [125, 60, -5, -255][r] + (s - 25000) // 1000
        jade_point += [0.5, 0.2, -0.2, -0.5][r]
        num_tops += 1 if r == 0 else 0
        num_games += 1

    print(f'# of games: {num_games}')
    print(f'Average ranking: {ranking / num_games + 1.0}')
    print(f'Average delta of grading points: {grading_point / num_games}')
    print(f'Average delta of jade points: {jade_point / num_games}')
    print(f'Top ratio: {num_tops / num_games}')


if __name__ == '__main__':
    main()
    sys.exit(0)
