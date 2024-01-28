#!/usr/bin/env python3

from argparse import ArgumentParser
import sys


def _parse(room_filter: int | None) -> None:
    for line in sys.stdin:
        columns = line.split('\t')
        if len(columns) != 8:
            raise RuntimeError(f'An invalid line: {line}')
        uuid, sparse, numeric, progression, _, _, _, result = columns

        sparse_fields = [int(x) for x in sparse.split(',')]

        room = sparse_fields[0]
        if room_filter is not None:
            if room < room_filter:
                continue
        game_style = sparse_fields[1] - 5
        grades = [None] * 4
        grades[0] = sparse_fields[2] - 7
        grades[1] = sparse_fields[3] - 23
        grades[2] = sparse_fields[4] - 39
        grades[3] = sparse_fields[5] - 55
        seat = sparse_fields[6] - 71
        chang = sparse_fields[7] - 75
        ju = sparse_fields[8] - 78
        num_left_tiles = sparse_fields[9] - 82

        if seat != ju or num_left_tiles != 69:
            continue

        numeric_fields = [int(x) for x in numeric.split(',')]

        benchang = numeric_fields[0]
        deposites = numeric_fields[1]
        bor_scores = numeric_fields[2:]

        progression_fields = [int(x) for x in progression.split(',')]
        if len(progression_fields) != 1:
            raise RuntimeError('A logic error.')

        result_fields = [int(x) for x in result.split(',')]
        round_score_deltas = result_fields[:4]
        eog_scores = result_fields[8:]

        print(
            f'{uuid}\t{room},{game_style + 5},{grades[0] + 7},{grades[1] + 23},{grades[2] + 39}'
            f',{grades[3] + 55},{chang + 71},{ju + 74}\t{benchang},{deposites},{bor_scores[0]}'
            f',{bor_scores[1]},{bor_scores[2]},{bor_scores[3]}\t{round_score_deltas[0]}'
            f',{round_score_deltas[1]},{round_score_deltas[2]},{round_score_deltas[3]}'
            f',{eog_scores[0]},{eog_scores[1]},{eog_scores[2]},{eog_scores[3]}')


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        '--filter-by-room', choices=('bronze', 'silver', 'gold', 'jade', 'throne'),
        help='filter annotations by the specified room or above')
    args = parser.parse_args()

    if args.filter_by_room is None:
        room_filter = None
    else:
        room_filter = {
            'bronze': 0, 'silver': 1, 'gold': 2, 'jade': 3, 'throne': 4
        }[args.filter_by_room]

    _parse(room_filter)


if __name__ == '__main__':
    _main()
