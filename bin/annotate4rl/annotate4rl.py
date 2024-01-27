#!/usr/bin/env python3

from pathlib import Path
from argparse import ArgumentParser
import sys


_AnnotationKey = tuple[str, int, int, int, int, int, int]
_GameKey = tuple[str, int]
_RoundKey = tuple[str, int, int, int, int]


def _get_annotation_key(uuid: str, sparse: str, numeric: str, progression: str) -> _AnnotationKey:
    sparse_fields = [int(x) for x in sparse.split(',')]
    numeric_fields = [int(x) for x in numeric.split(',')]
    progression_fields = [int(x) for x in progression.split(',')]

    seat = sparse_fields[6] - 71
    assert 0 <= seat and seat < 4
    chang = sparse_fields[7] - 75
    assert 0 <= chang and chang < 3
    ju = sparse_fields[8] - 78 # pylint: disable=invalid-name
    assert 0 <= ju and ju < 4
    num_drawn_tiles = sparse_fields[9] - 82
    assert 0 <= num_drawn_tiles and num_drawn_tiles <= 69

    ben = numeric_fields[0]
    assert ben >= 0

    turn = len(progression_fields)

    return uuid, seat, chang, ju, ben, turn, num_drawn_tiles


def _get_keys(uuid: str, sparse: str, numeric: str) -> tuple[_GameKey, _RoundKey]:
    sparse_fields = [int(x) for x in sparse.split(',')]
    numeric_fields = [int(x) for x in numeric.split(',')]

    seat = sparse_fields[6] - 71
    assert 0 <= seat and seat < 4
    chang = sparse_fields[7] - 75
    assert 0 <= chang and chang < 3
    ju = sparse_fields[8] - 78 # pylint: disable=invalid-name
    assert 0 <= ju and ju < 4

    ben = numeric_fields[0]
    assert ben >= 0

    return (uuid, seat), (uuid, seat, chang, ju, ben)


def _parse(
        input_lines: list[str], room_filter: int | None, grade_filter: int | None) -> None:
    annotations: list[tuple[_AnnotationKey, str]] = []
    for line in input_lines:
        line = line.rstrip('\n')
        columns = line.split('\t')
        if len(columns) != 8:
            raise RuntimeError(f'An invalid line: {line}')
        uuid, sparse, numeric, progression, _, _, _, _ = columns

        sparse_fields = [int(x) for x in sparse.split(',')]
        room = sparse_fields[0]
        seat = sparse_fields[6] - 71
        grade = sparse_fields[2 + seat] - (7 + 16 * seat)
        assert 0 <= grade and grade <= 15

        if room_filter is not None and room < room_filter:
            continue

        if grade_filter is not None and grade < grade_filter:
            continue

        annotation_key = _get_annotation_key(uuid, sparse, numeric, progression)

        annotation = (annotation_key, line)
        annotations.append(annotation)
    annotations.sort(key=lambda e: e[0])

    if len(annotations) == 0:
        return

    i = 0
    while i < len(annotations):
        line = annotations[i][1]
        columns = line.split('\t')
        uuid, sparse, numeric, progression, candidates, action, round_summary, results = columns
        results = ','.join(results.split(',')[:8])

        game_key, round_key = _get_keys(uuid, sparse, numeric)

        if i + 1 < len(annotations):
            next_line = annotations[i + 1][1]
            next_columns = next_line.split('\t')
            next_uuid, next_sparse, next_numeric, next_progression, next_candidates, _, _, _ = next_columns

            next_game_key, next_round_key = _get_keys(next_uuid, next_sparse, next_numeric)

            if game_key == next_game_key:
                end_of_round = (round_key != next_round_key)
                if end_of_round:
                    print(
                        f'{uuid}\t{sparse}\t{numeric}\t{progression}\t{candidates}\t{action}'
                        f'\t{next_sparse}\t{next_numeric}\t{next_progression}\t{next_candidates}'
                        f'\t{round_summary}\t{results}')
                else:
                    print(
                        f'{uuid}\t{sparse}\t{numeric}\t{progression}\t{candidates}\t{action}'
                        f'\t{next_sparse}\t{next_numeric}\t{next_progression}\t{next_candidates}')
                i += 1
                continue

        print(
            f'{uuid}\t{sparse}\t{numeric}\t{progression}\t{candidates}\t{action}\t{round_summary}'
            f'\t{results}')
        i += 1


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        '--filter-by-room', choices=('bronze', 'silver', 'gold', 'jade', 'throne'), default='gold',
        help='filter annotations by the specified room or above')
    parser.add_argument(
        '--filter-by-grade',
        choices=('expert1', 'expert2', 'expert3', 'master1', 'master2', 'master3', 'saint1', 'saint2', 'saint3', 'celestial'),
        help='filter annotation by the specified grade or above')
    parser.add_argument('INPUT', nargs='?', type=Path, help='path to input file')
    args = parser.parse_args()

    if args.filter_by_room is None:
        room_filter = None
    else:
        room_filter = {
            'bronze': 0, 'silver': 1, 'gold': 2, 'jade': 3, 'throne': 4
        }[args.filter_by_room]

    if args.filter_by_grade is None:
        grade_filter = None
    else:
        grade_filter = {
            'expert1': 6, 'expert2': 7, 'expert3': 8, 'master1': 9, 'master2': 10, 'master3': 11,
            'saint1': 12, 'saint2': 13, 'saint3': 14, 'celestial': 15
        }[args.filter_by_grade]

    input_path: Path | None = args.INPUT

    if str(input_path) == '-':
        input_path = None

    if input_path is None:
        input_lines = sys.stdin.readlines()
    else:
        if not input_path.exists():
            raise RuntimeError(f'{input_path}: Does not exist.')
        if not input_path.is_file():
            raise RuntimeError(f'{input_path}: Not a file.')
        with open(input_path, encoding='UTF-8') as fp: # pylint: disable=invalid-name
            input_lines = fp.readlines()

    _parse(input_lines, room_filter, grade_filter)


if __name__ == '__main__':
    _main()
