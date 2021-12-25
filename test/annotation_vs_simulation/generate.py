#!/usr/bin/env python3

from pathlib import Path
from argparse import ArgumentParser
import json
from typing import Tuple
import sys


def load_paishan(paishan_path: Path) -> dict:
    paishan_map = {}

    with open(paishan_path) as f:
        for line in f:
            line = line.rstrip('\n')
            uuid, chang, ju, ben, paishan = line.split('\t')
            chang = int(chang)
            ju = int(ju)
            ben = int(ben)
            paishan = json.loads('[' + paishan + ']')
            if len(paishan) != 136:
                raise RuntimeError(len(paishan))
            paishan_map[(uuid, chang, ju, ben)] = paishan

    return paishan_map


def convert_annotation(paishan_map: dict, annotation_path: Path) -> dict:
    game_map = {}

    with open(annotation_path) as f:
        for (i, line) in enumerate(f):
            i += 1
            line = line.rstrip('\n')
            uuid, sparse, numeric, progression, candidates, index, result = line.split('\t')
            sparse = json.loads('[' + sparse + ']')
            numeric = json.loads('[' + numeric + ']')
            progression = json.loads('[' + progression + ']')
            candidates = json.loads('[' + candidates + ']')
            index = int(index)
            if index >= len(candidates):
                raise RuntimeError(index)
            result = json.loads('[' + result + ']')

            room = sparse[0]
            style = sparse[1] - 5
            seat = sparse[2] - 7
            chang = sparse[3] - 11
            ju = sparse[4] - 14
            ben = numeric[0]

            if uuid not in game_map:
                game_map[uuid] = {}

            if 'room' not in game_map[uuid]:
                game_map[uuid]['room'] = room
            elif game_map[uuid]['room'] != room:
                raise RuntimeError(
                    f'{annotation_path}:{i}: An inconsistent annotation.')

            if 'style' not in game_map[uuid]:
                game_map[uuid]['style'] = style
            elif game_map[uuid]['style'] != style:
                raise RuntimeError(
                    f'{annotation_path}:{i}: An inconsistent annotation.')

            if 'players' not in game_map[uuid]:
                game_map[uuid]['players'] = [{}, {}, {}, {}]

            if 'grade' not in game_map[uuid]['players'][seat]:
                for x in sparse:
                    if 273 <= x and x <= 288:
                        game_map[uuid]['players'][seat]['grade'] = x - 273
            else:
                for x in sparse:
                    if 273 <= x and x <= 288 and game_map[uuid]['players'][seat]['grade'] != x - 273:
                        raise RuntimeError(
                            f'{annotation_path}:{i}: An inconsistent annotation.')

            if 'final_ranking' not in game_map[uuid]['players'][seat]:
                game_map[uuid]['players'][seat]['final_ranking'] = result[9]
            elif game_map[uuid]['players'][seat]['final_ranking'] != result[9]:
                raise RuntimeError(
                    f'{annotation_path}:{i}: An inconsistent annotation.')

            if 'final_score' not in game_map[uuid]['players'][seat]:
                game_map[uuid]['players'][seat]['final_score'] = result[10]
            elif game_map[uuid]['players'][seat]['final_score'] != result[10]:
                raise RuntimeError(
                    f'{annotation_path}:{i}: An inconsistent annotation.')

            if 'rounds' not in game_map[uuid]:
                game_map[uuid]['rounds'] = {}

            if (chang, ju, ben) not in game_map[uuid]['rounds']:
                game_map[uuid]['rounds'][(chang, ju, ben)] = {}

            if 'paishan' not in game_map[uuid]['rounds'][(chang, ju, ben)]:
                game_map[uuid]['rounds'][(chang, ju, ben)]['paishan'] = paishan_map[(uuid, chang, ju, ben)]

            if 'decisions' not in game_map[uuid]['rounds'][(chang, ju, ben)]:
                game_map[uuid]['rounds'][(chang, ju, ben)]['decisions'] = []

            game_map[uuid]['rounds'][(chang, ju, ben)]['decisions'].append({
                'sparse': sparse,
                'numeric': numeric,
                'progression': progression,
                'candidates': candidates,
                'index': index
            })

            if progression == [0]:
                delta_scores = result[1:5]
                delta_scores = [delta_scores[i] for i in range(-ju, -ju + 4)]
                if 'delta_scores' not in game_map[uuid]['rounds'][(chang, ju, ben)]:
                    game_map[uuid]['rounds'][(chang, ju, ben)]['delta_scores'] = delta_scores
                elif game_map[uuid]['rounds'][(chang, ju, ben)]['delta_scores'] != delta_scores:
                    raise RuntimeError(
                        f'{annotation_path}:{i}:\
 {game_map[uuid]["rounds"][(chang, ju, ben)]["delta_scores"]} != {delta_scores}')

    return game_map


def decision_order_key(decision: dict) -> Tuple[int, int]:
    k0 = len(decision['progression'])
    k1 = None
    for s in decision['sparse']:
        if 203 <= s and s <= 272:
            k1 = -s
            break
    if k1 is None:
        raise RuntimeError(decision['sparse'])
    k2 = -1
    for candidate in decision['candidates']:
        if 222 <= candidate and candidate <= 311:
            k2 = 0
            break
        if 312 <= candidate and candidate <= 431:
            relseat = (candidate - 312) // 40
            k2 = 2 - relseat
            break
        if 432 <= candidate and candidate <= 542:
            relseat = (candidate - 432) // 37
            k2 = 2 - relseat
            break
        if 543 <= candidate and candidate <= 545:
            relseat = candidate - 543
            k2 = 2 - relseat
            break
    return (k0, k1, k2)


def unique(l: list) -> None:
    if len(l) == 1:
        return
    i = 0
    j = 1
    while j < len(l):
        x = l[i]
        y = l[j]
        if x == y:
            l.pop(j)
            continue
        i += 1
        j += 1


def main():
    parser = ArgumentParser(description='Generate test cases.')

    parser.add_argument(
        '--paishan-path', type=Path, required=True,
        help='path to paishan data file')
    parser.add_argument(
        '--annotation-path', type=Path, required=True,
        help='path to annotation file')
    parser.add_argument(
        '--output-prefix', type=Path, required=True,
        help='prefix of test case files to generate')
    args = parser.parse_args()

    paishan_path = args.paishan_path
    if not paishan_path.exists():
        raise RuntimeError(f'{paishan_path}: Does not exist.')
    if not paishan_path.is_file():
        raise RuntimeError(f'{paishan_path}: Not a file.')

    annotation_path = args.annotation_path
    if not annotation_path.exists():
        raise RuntimeError(f'{annotation_path}: Does not exist')
    if not annotation_path.is_file():
        raise RuntimeError(f'{annotation_path}: Not a file.')

    paishan_map = load_paishan(paishan_path)

    game_map = convert_annotation(paishan_map, annotation_path)

    for uuid, value in game_map.items():
        record = {
            'uuid': uuid,
            'room': value['room'],
            'style': value['style'],
            'players': value['players'],
            'rounds': []
        }
        for (chang, ju, ben), v in value['rounds'].items():
            record['rounds'].append({
                'chang': chang,
                'ju': ju,
                'ben': ben,
                'paishan': v['paishan'],
                'decisions': v['decisions'],
                'delta_scores': v['delta_scores']
            })

        record['rounds'].sort(key=lambda e: (e['chang'], e['ju'], e['ben']))
        for r in record['rounds']:
            r['decisions'].sort(key=decision_order_key)
            # ダブロン・トリプルロンの場合，各アノテーションが
            # 2重・3重に重複しているので，重複を除去する．
            unique(r['decisions'])

        if not args.output_prefix.exists():
            args.output_prefix.mkdir(parents=True)
        if not args.output_prefix.is_dir():
            raise RuntimeError(f'{args.output_prefix}: Not a directory.')

        path = args.output_prefix / f'{uuid}.json'
        if path.exists():
            raise RuntimeError(f'{path}: File already exists.')

        with open(path, 'w') as f:
            json.dump(record, f, separators=(',', ':'))


if __name__ == '__main__':
    main()
    sys.exit(0)
