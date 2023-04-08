#!/usr/bin/env python3

import re
import datetime
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Callable
import sys
import torch
from torch import backends
from torch import nn
from kanachan.simulation import simulate


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Run Mahjong Soul game simulation.')
    parser.add_argument('--device', help='device', metavar='DEV')
    parser.add_argument(
        '--dtype', choices=('half', 'float16', 'float', 'float32', 'double', 'float64'),
        help='data type (defaults to `float16`)', metavar='DTYPE')
    parser.add_argument(
        '--baseline-model', type=Path, required=True,
        help='path to config file for baseline model', metavar='BASELINE_MODEL')
    parser.add_argument(
        '--baseline-grade', type=int, required=True,
        help='grade of baseline model', metavar='BASELINE_GRADE')
    parser.add_argument(
        '--proposed-model', type=Path, required=True,
        help='path to config file for proposed model', metavar='PROPOSED_MODEL')
    parser.add_argument(
        '--proposed-grade', type=int, required=True,
        help='grade of proposed model', metavar='PROPOSED_GRADE')
    parser.add_argument(
        '--room', choices=('bronze', 'silver', 'gold', 'jade', 'throne',),
        required=True, help='room for simulation')
    parser.add_argument(
        '--dongfengzhan', action='store_true',
        help='simulate dong feng zhan (東風戦)')
    parser.add_argument(
        '--mode', default='2vs2', choices=('2vs2', '1vs3',),
        help='simulation mode (defaults to `2vs2`)')
    parser.add_argument(
        '--non-duplicated', action='store_true',
        help='disable duplicated simulation')
    parser.add_argument(
        '-n', default=1, type=int,
        help='# of sets to simulate (defaults to `1`)',
        metavar='N')
    parser.add_argument('--batch-size', default=1, type=int, metavar='N_BATCH')
    parser.add_argument('--concurrency', type=int, metavar='N_THREADS')

    return parser.parse_args()


def _main():
    config = _parse_arguments()

    if config.device is not None:
        match = re.search('^(?:cpu|cuda(?::\\d+)?)$', config.device)
        if match is None:
            raise RuntimeError(f'{config.device}: invalid device')
        device = config.device
    elif backends.cuda.is_built():
        device = 'cuda'
    else:
        device = 'cpu'

    if config.dtype is None:
        if device == 'cpu':
            config.dtype = 'float32'
        else:
            config.dtype = 'float16'
    if config.dtype in ('half', 'float16'):
        dtype = torch.half
    elif config.dtype in ('float', 'float32'):
        dtype = torch.float
    elif config.dtype in ('double', 'float64'):
        dtype = torch.double
    else:
        raise ValueError(f'{config.dtype}: An invalid data type.')

    if not config.baseline_model.exists():
        raise RuntimeError(f'{config.baseline_model}: Does not exist.')
    if not config.baseline_model.is_file():
        raise RuntimeError(f'{config.baseline_model}: Not a file.')
    baseline_model: nn.Module = torch.load(config.baseline_model, map_location='cpu')
    baseline_model.to(device=device, dtype=dtype)
    baseline_model.eval()

    if config.baseline_grade < 0 or 15 < config.baseline_grade:
        raise RuntimeError(
            f'{config.baseline_grade}: An invalid value for `--baseline-grade`.')

    if not config.proposed_model.exists():
        raise RuntimeError(f'{config.proposed_model}: Does not exist.')
    if not config.proposed_model.is_file():
        raise RuntimeError(f'{config.proposed_model}: Not a file.')
    proposed_model: nn.Module = torch.load(config.proposed_model, map_location='cpu')
    proposed_model.to(device=device, dtype=dtype)
    proposed_model.eval()

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

    if config.batch_size < 1:
        raise RuntimeError(f'{config.batch_size}: An invalid value for the `--batch-size` option.')

    if config.concurrency is None:
        config.concurrency = config.batch_size * 2
    if config.concurrency < 1:
        raise RuntimeError(
            f'{config.concurrency}: An invalid value for the `--concurrency` option.')

    with torch.no_grad():
        start_time = datetime.datetime.now()
        results = simulate(
            device, dtype, config.baseline_grade, baseline_model,
            config.proposed_grade, proposed_model, mode, config.n,
            config.batch_size, config.concurrency)

    elapsed_time = datetime.datetime.now() - start_time
    if config.non_duplicated:
        print(f'Elapsed time: {elapsed_time} ({elapsed_time / config.n}/game)')
    elif config.mode == '2vs2':
        print(f'Elapsed time: {elapsed_time} ({elapsed_time / (config.n * 6.0)}/game)')
    else:
        assert config.mode == '1vs3'
        print(f'Elapsed time: {elapsed_time} ({elapsed_time / (config.n * 4.0)}/game)')

    num_games = len(results)

    def get_grading_point(ranking: int, score: int) -> int:
        return [125, 60, -5, -255][ranking] + (score - 25000) // 1000

    def get_soul_point(ranking: int, _: int) -> float:
        return [0.5, 0.2, -0.2, -0.5][ranking]

    Statistic = Tuple[float, float]

    def get_statistic(
            results: List[object], proposed: int,
            callback: Callable[[int, int], float]) -> Statistic:
        assert proposed in (0, 1)
        average = 0.0
        num_proposed = 0
        for game in results:
            assert len(game['proposed']) == 4
            assert len(game['ranking']) == 4
            for i in range(4):
                assert game['proposed'][i] in (0, 1)
                ranking = game['ranking'][i]
                score = game['scores'][i]
                if game['proposed'][i] == proposed:
                    average += callback(ranking, score)
                    num_proposed += 1
        assert num_proposed >= 1
        average /= num_proposed

        variance = 0.0
        num_proposed = 0
        for game in results:
            for i in range(4):
                ranking = game['ranking'][i]
                score = game['scores'][i]
                if game['proposed'][i] == proposed:
                    variance += (callback(ranking, score) - average) ** 2.0
                    num_proposed += 1
        assert num_proposed >= 1
        # Unbiased sample variance.
        variance /= (num_proposed - 1)

        return average, variance

    Statistics = Tuple[Statistic, Statistic, Statistic, Statistic, Statistic]

    def get_statistics(results: List[object], proposed: int) -> Statistics:
        ranking_statistic = get_statistic(results, proposed, lambda r, s: r)
        grading_point_statistic = get_statistic(
            results, proposed, get_grading_point)
        soul_point_statistic = get_statistic(results, proposed, get_soul_point)

        top_rate = 0.0
        num_proposed = 0
        for game in results:
            for i in range(4):
                if game['proposed'][i] == proposed:
                    num_proposed += 1
                    if game['ranking'][i] == 0:
                        top_rate += 1.0
        top_rate /= num_proposed
        # Unbiased sample variance.
        top_rate_variance = top_rate * (1.0 - top_rate) / (num_proposed - 1)

        quinella_rate = 0.0
        num_proposed = 0
        for game in results:
            for i in range(4):
                if game['proposed'][i] == proposed:
                    num_proposed += 1
                    if game['ranking'][i] <= 1:
                        quinella_rate += 1.0
        quinella_rate /= num_proposed
        # Unbiased sample variance.
        quinella_rate_variance \
            = quinella_rate * (1.0 - quinella_rate) / (num_proposed - 1)

        return (
            ranking_statistic,
            grading_point_statistic,
            soul_point_statistic,
            (top_rate, top_rate_variance),
            (quinella_rate, quinella_rate_variance),)

    baseline_statistics = get_statistics(results, 0)
    proposed_statistics = get_statistics(results, 1)

    ranking_diff_average = 0.0
    for game in results:
        num_baseline = 0
        baseline_ranking = 0.0
        num_proposed = 0
        proposed_ranking = 0.0
        for i in range(4):
            ranking = game['ranking'][i]
            if game['proposed'][i] == 0:
                num_baseline += 1
                baseline_ranking += ranking
            else:
                assert game['proposed'][i] == 1
                num_proposed += 1
                proposed_ranking += ranking
        assert num_baseline >= 1
        assert num_proposed >= 1
        baseline_ranking /= num_baseline
        proposed_ranking /= num_proposed
        diff = proposed_ranking - baseline_ranking
        ranking_diff_average += diff
    ranking_diff_average /= num_games

    ranking_diff_variance = 0.0
    for game in results:
        num_baseline = 0
        baseline_ranking = 0.0
        num_proposed = 0
        proposed_ranking = 0.0
        for i in range(4):
            ranking = game['ranking'][i]
            if game['proposed'][i] == 0:
                num_baseline += 1
                baseline_ranking += ranking
            else:
                assert game['proposed'][i] == 1
                num_proposed += 1
                proposed_ranking += ranking
        assert num_baseline >= 1
        assert num_proposed >= 1
        baseline_ranking /= num_baseline
        proposed_ranking /= num_proposed
        diff = proposed_ranking - baseline_ranking
        ranking_diff_variance += (diff - ranking_diff_average) ** 2.0
    # Unbiased sample variance.
    ranking_diff_variance /= (num_games - 1)

    print(f'''---
num_games: {num_games}
baseline:
  ranking:
    average: {baseline_statistics[0][0] + 1.0}
    variance: {baseline_statistics[0][1]}
  grading_point:
    average: {baseline_statistics[1][0]}
    variance: {baseline_statistics[1][1]}
  soul_point:
    average: {baseline_statistics[2][0]}
    variance: {baseline_statistics[2][1]}
  top_rate:
    average: {baseline_statistics[3][0]}
    variance: {baseline_statistics[3][1]}
  quinella_rate:
    average: {baseline_statistics[4][0]}
    variance: {baseline_statistics[4][1]}
proposed:
  ranking:
    average: {proposed_statistics[0][0] + 1.0}
    variance: {proposed_statistics[0][1]}
  grading_point:
    average: {proposed_statistics[1][0]}
    variance: {proposed_statistics[1][1]}
  soul_point:
    average: {proposed_statistics[2][0]}
    variance: {proposed_statistics[2][1]}
  top_rate:
    average: {proposed_statistics[3][0]}
    variance: {proposed_statistics[3][1]}
  quinella_rate:
    average: {proposed_statistics[4][0]}
    variance: {proposed_statistics[4][1]}
difference:
  ranking:
    average: {ranking_diff_average}
    variance: {ranking_diff_variance}
''')


if __name__ == '__main__':
    _main()
    sys.exit(0)
