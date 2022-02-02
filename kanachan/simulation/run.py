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
from kanachan.training.bert.model_mode import ModelMode
from kanachan.simulation import simulate


_MODEL_CONFIG_SCHEMA = {
    'type': 'object',
    'required': [
        'encoder',
        'decoder',
        'model',
        'grade'
    ],
    'properties': {
        'encoder': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs',
                'snapshot'
            ],
            'properties': {
                'module': {
                    'type': 'string'
                },
                'class': {
                    'type': 'string'
                },
                'kwargs': {
                    'type': 'object'
                },
                'snapshot': {
                    'type': 'string'
                }
            },
            'additionalProperties': False
        },
        'decoder': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs',
                'snapshot'
            ],
            'properties': {
                'module': {
                    'type': 'string'
                },
                'class': {
                    'type': 'string'
                },
                'kwargs': {
                    'type': 'object'
                },
                'snapshot': {
                    'type': 'string'
                }
            },
            'additionalProperties': False
        },
        'model': {
            'type': 'object',
            'required': [
                'module',
                'class'
            ],
            'properties': {
                'module': {
                    'type': 'string'
                },
                'class': {
                    'type': 'string'
                }
            },
            'additionalProperties': False
        },
        'grade': {
            'type': 'integer',
            'minimum': 0,
            'maximum': 15
        }
    },
    'additionalProperties': False
}


def main():
    ap = ArgumentParser(description='Run Mahjong Soul game simulation.')
    ap.add_argument('--device', help='device', metavar='DEV')
    ap.add_argument(
        '--dtype', default='float16', choices=('float16', 'float32',),
        help='data type (defaults to `float16`)', metavar='DTYPE')
    ap.add_argument(
        '--baseline-model-config', type=Path, required=True,
        help='path to config file for baseline model',
        metavar='BASELINE_MODEL_CONFIG')
    ap.add_argument(
        '--proposed-model-config', type=Path, required=True,
        help='path to config file for proposed model',
        metavar='PROPOSED_MODEL_CONFIG')
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

    if not config.baseline_model_config.exists():
        raise RuntimeError(f'{config.baseline_model_config}: Does not exist.')
    if not config.baseline_model_config.is_file():
        raise RuntimeError(f'{config.baseline_model_config}: Not a file.')
    with open(config.baseline_model_config) as f:
        baseline_model_config = yaml.load(f, Loader=yaml.Loader)
    jsonschema.validate(
        instance=baseline_model_config, schema=_MODEL_CONFIG_SCHEMA)

    if not config.proposed_model_config.exists():
        raise RuntimeError(f'{config.proposed_model_config}: Does not exist.')
    if not config.proposed_model_config.is_file():
        raise RuntimeError(f'{config.proposed_model_config}: Not a file.')
    with open(config.proposed_model_config) as f:
        proposed_model_config = yaml.load(f, Loader=yaml.Loader)
    jsonschema.validate(
        instance=proposed_model_config, schema=_MODEL_CONFIG_SCHEMA)

    baseline_encoder_module = baseline_model_config['encoder']['module']
    baseline_encoder_module = importlib.import_module(baseline_encoder_module)
    baseline_encoder_class = baseline_model_config['encoder']['class']
    baseline_encoder_class = getattr(
        baseline_encoder_module, baseline_encoder_class)
    baseline_encoder_kwargs = baseline_model_config['encoder']['kwargs']
    baseline_encoder = baseline_encoder_class(**baseline_encoder_kwargs)

    baseline_decoder_module = baseline_model_config['decoder']['module']
    baseline_decoder_module = importlib.import_module(baseline_decoder_module)
    baseline_decoder_class = baseline_model_config['decoder']['class']
    baseline_decoder_class = getattr(
        baseline_decoder_module, baseline_decoder_class)
    baseline_decoder_kwargs = baseline_model_config['decoder']['kwargs']
    baseline_decoder = baseline_decoder_class(**baseline_decoder_kwargs)

    baseline_model_module = baseline_model_config['model']['module']
    baseline_model_module = importlib.import_module(baseline_model_module)
    baseline_model_class = baseline_model_config['model']['class']
    baseline_model_class = getattr(baseline_model_module, baseline_model_class)
    baseline_model = baseline_model_class(baseline_encoder, baseline_decoder)
    baseline_model.to(device=device, dtype=dtype)

    baseline_encoder_snapshot = Path(
        baseline_model_config['encoder']['snapshot'])
    if not baseline_encoder_snapshot.exists():
        raise RuntimeError(f'{baseline_encoder_snapshot}: Does not exist.')
    if not baseline_encoder_snapshot.is_file():
        raise RuntimeError(f'{baseline_encoder_snapshot}: Not a file.')
    baseline_encoder.load_state_dict(torch.load(baseline_encoder_snapshot))

    baseline_decoder_snapshot = Path(
        baseline_model_config['decoder']['snapshot'])
    if not baseline_decoder_snapshot.exists():
        raise RuntimeError(f'{baseline_decoder_snapshot}: Does not exist.')
    if not baseline_decoder_snapshot.is_file():
        raise RuntimeError(f'{baseline_decoder_snapshot}: Not a file.')
    baseline_decoder.load_state_dict(torch.load(baseline_decoder_snapshot))

    baseline_model_grade = baseline_model_config['grade']

    proposed_encoder_module = proposed_model_config['encoder']['module']
    proposed_encoder_module = importlib.import_module(proposed_encoder_module)
    proposed_encoder_class = proposed_model_config['encoder']['class']
    proposed_encoder_class = getattr(
        proposed_encoder_module, proposed_encoder_class)
    proposed_encoder_kwargs = proposed_model_config['encoder']['kwargs']
    proposed_encoder = proposed_encoder_class(**proposed_encoder_kwargs)

    proposed_decoder_module = proposed_model_config['decoder']['module']
    proposed_decoder_module = importlib.import_module(proposed_decoder_module)
    proposed_decoder_class = proposed_model_config['decoder']['class']
    proposed_decoder_class = getattr(
        proposed_decoder_module, proposed_decoder_class)
    proposed_decoder_kwargs = proposed_model_config['decoder']['kwargs']
    proposed_decoder = proposed_decoder_class(**proposed_decoder_kwargs)

    proposed_model_module = proposed_model_config['model']['module']
    proposed_model_module = importlib.import_module(proposed_model_module)
    proposed_model_class = proposed_model_config['model']['class']
    proposed_model_class = getattr(proposed_model_module, proposed_model_class)
    proposed_model = proposed_model_class(proposed_encoder, proposed_decoder)
    proposed_model.to(device=device, dtype=dtype)

    proposed_encoder_snapshot = Path(
        proposed_model_config['encoder']['snapshot'])
    if not proposed_encoder_snapshot.exists():
        raise RuntimeError(f'{proposed_encoder_snapshot}: Does not exist.')
    if not proposed_encoder_snapshot.is_file():
        raise RuntimeError(f'{proposed_encoder_snapshot}: Not a file.')
    proposed_encoder.load_state_dict(torch.load(proposed_encoder_snapshot))

    proposed_decoder_snapshot = Path(
        proposed_model_config['decoder']['snapshot'])
    if not proposed_decoder_snapshot.exists():
        raise RuntimeError(f'{proposed_decoder_snapshot}: Does not exist.')
    if not proposed_decoder_snapshot.is_file():
        raise RuntimeError(f'{proposed_decoder_snapshot}: Not a file.')
    proposed_decoder.load_state_dict(torch.load(proposed_decoder_snapshot))

    proposed_model_grade = proposed_model_config['grade']

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
                device, dtype, mode, baseline_model_grade, baseline_model,
                proposed_model_grade, proposed_model, result)

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
