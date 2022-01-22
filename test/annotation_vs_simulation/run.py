#!/usr/bin/env python3

from pathlib import Path
import os
import json
import sys
from kanachan.simulation import (TestModel, test,)


def _on_leaf(test_file_path: Path):
    print(f'{test_file_path}: ', end='', flush=True)

    with open(test_file_path) as f:
        test_data = json.load(f)

    uuid = test_data['uuid']
    room = test_data['room']
    style = test_data['style']
    players = test_data['players']
    rounds = test_data['rounds']

    player_grades = tuple(p['grade'] for p in players)

    paishan_list = []
    for r in rounds:
        paishan = r['paishan']
        paishan_list.append(paishan)

    simulation_mode = 1 # Non-duplicated simulation mode
    if style == 0:
        simulation_mode += 2 # Dong Feng Zhan (東風戦)
    simulation_mode += (1 << (3 + room))

    test_model = TestModel(rounds)

    result = test(simulation_mode, player_grades, test_model, paishan_list)

    if len(result['rounds']) != len(test_data['rounds']):
        raise RuntimeError(f'{result["rounds"]} != {test_data["rounds"]}')

    for i in range(len(test_data['rounds'])):
        round_data = test_data['rounds'][i]
        round_result = result['rounds'][i]
        for j in range(4):
            delta_score_to_be = round_data['delta_scores'][j]
            delta_score_as_is = round_result[j]['delta_score']
            if delta_score_as_is != delta_score_to_be:
                delta_scores_as_is = [round_result[k]['delta_score'] for k in range(4)]
                raise RuntimeError(
                    f'''{i}-th round:
{delta_scores_as_is}
{round_data["delta_scores"]}''')

    final_ranking = [test_data['players'][i]['final_ranking'] for i in range(4)]
    if result['final_ranking'] != final_ranking:
        raise RuntimeError(f'{result["final_ranking"]} != {final_ranking}')

    final_scores = [test_data['players'][i]['final_score'] for i in range(4)]
    if result['final_scores'] != final_scores:
        raise RuntimeError(f'{result["final_scores"]} != {final_scores}')

    print('PASS')


def main():
    if len(sys.argv) < 2:
        raise RuntimeError('Too few arguments.')
    if len(sys.argv) > 3:
        raise RuntimeError('Too many arguments.')

    path = Path(sys.argv[1])
    if not path.exists():
        raise RuntimeError(f'{path}: Does not exist.')

    if path.is_file():
        _on_leaf(path)
        return

    skip_list_path = None
    if len(sys.argv) == 3:
        skip_list_path = Path(sys.argv[2])
        if not skip_list_path.is_file():
            raise RuntimeError(f'{skip_list_path}: Not a file.')

    skip_list = set()
    if skip_list_path is not None:
        with open(skip_list_path) as f:
            for line in f:
                line = line.rstrip('\n')
                skip_list.add(line)

    if not path.is_dir():
        raise RuntimeError(f'{path}: Neither a file nor a directory.')
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if not filename.endswith('.json'):
                continue
            test_file_path = Path(dirpath) / filename
            if test_file_path.stem in skip_list:
                print(f'{test_file_path}: SKIP')
                continue
            _on_leaf(test_file_path)


if __name__ == '__main__':
    main()
    sys.exit(0)
