#!/usr/bin/env python3

from pathlib import Path
import random
from argparse import ArgumentParser
from typing import Optional, Tuple, List
import sys


_AnnotationKey = Tuple[str, int, int, int, int, int, int]


def _get_annotation_key(uuid: str, sparse: str, numeric: str, progression: str) -> _AnnotationKey:
    sparse_fields = [int(field) for field in sparse.split(',')]
    numeric_fields = [int(field) for field in numeric.split(',')]
    progression_fields = [int(field) for field in progression.split(',')]

    seat = sparse_fields[2] - 7
    chang = sparse_fields[3] - 11
    ju = sparse_fields[4] - 14 # pylint: disable=invalid-name
    num_drawn_tiles = None
    for field in sparse_fields:
        if 203 <= field and field <= 272:
            num_drawn_tiles = 272 - field
            break
    assert num_drawn_tiles is not None

    ben = numeric_fields[0]

    turn = len(progression_fields)

    return uuid, seat, chang, ju, ben, turn, num_drawn_tiles


def _parse(input_lines: List[str], room_filter: int, td_aware: bool, curriculum: bool) -> None:
    AnnotationValue = Tuple[str, str, str, str, str, str]
    annotations: List[Tuple[_AnnotationKey, AnnotationValue]] = []
    for line in input_lines:
        columns = line.split('\t')
        if len(columns) != 7:
            raise RuntimeError(f'An invalid line: {line}')
        uuid, sparse, numeric, progression, actions, index, results = columns

        sparse_fields = [int(field) for field in sparse.split(',')]
        room = sparse_fields[0]
        if room < room_filter:
            continue

        annotation_key = _get_annotation_key(uuid, sparse, numeric, progression)

        annotation = (annotation_key, (sparse, numeric, progression, actions, index, results))
        annotations.append(annotation)
    annotations.sort(key=lambda e: e[0])

    if len(annotations) == 0:
        return

    output_line_chunks: List[List[str]] = []
    output_line_chunks.append([])

    i = 0
    while i < len(annotations):
        prev = annotations[i]
        prev_uuid = prev[0][0]
        prev_seat = prev[0][1]

        next_uuid = None
        next_seat = None
        if i + 1 < len(annotations):
            _next = annotations[i + 1]
            next_uuid = _next[0][0]
            next_seat = _next[0][1]

        prev_sparse = prev[1][0]
        prev_numeric = prev[1][1]
        prev_progression = prev[1][2]
        prev_actions = prev[1][3]
        prev_index = prev[1][4]

        if i + 1 >= len(annotations) or prev_uuid != next_uuid or prev_seat != next_seat:
            result_fields = [int(field) for field in prev[1][5].split(',')]
            game_rank = result_fields[9]
            game_score = result_fields[10]
            line = f'{prev_uuid}\t' if td_aware else ''
            line += f'{prev_sparse}\t{prev_numeric}\t{prev_progression}\t{prev_actions}\t{prev_index}\t{game_rank}\t{game_score}'
            output_line_chunks[-1].append(line)
            if i + 1 >= len(annotations):
                break
            output_line_chunks.append([])
            i += 1
            continue

        next_sparse = _next[1][0]
        next_numeric = _next[1][1]
        next_progression = _next[1][2]
        next_actions = _next[1][3]

        line = f'{prev_uuid}\t' if td_aware else ''
        line += f'{prev_sparse}\t{prev_numeric}\t{prev_progression}\t{prev_actions}\t{prev_index}\t{next_sparse}\t{next_numeric}\t{next_progression}\t{next_actions}'
        output_line_chunks[-1].append(line)
        i += 1

    if td_aware:
        for i, output_lines in enumerate(output_line_chunks):
            uuid = None
            sorted_output_lines: List[Tuple[_AnnotationKey, str]] = []
            for output_line in output_lines:
                uuid, sparse, numeric, progression, tail = output_line.split('\t', 4)
                annotation_key = _get_annotation_key(uuid, sparse, numeric, progression)
                output_line = '\t'.join((sparse, numeric, progression, tail))
                sorted_output_lines.append((annotation_key, output_line))
            sorted_output_lines.sort(key=lambda x: x[0], reverse=True)
            output_line_chunks[i] = [line[1] for line in sorted_output_lines]
        random.shuffle(output_line_chunks)

    if curriculum:
        for output_lines in output_line_chunks:
            num_lines = len(output_lines)
            for i, output_line in enumerate(output_lines):
                print(f'{num_lines - i - 1}\t{output_line}')
    else:
        for output_lines in output_line_chunks:
            for output_line in output_lines:
                print(output_line)


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        '--filter-by-room', choices=('bronze', 'silver', 'gold', 'jade', 'throne'), default='gold',
        help='filter annotations by the specified room or above')
    parser.add_argument('--td-aware', action='store_true', help='make output TD-aware')
    parser.add_argument('--curriculum', action='store_true', help='add curriculum index')
    parser.add_argument('INPUT', nargs='?', type=Path, help='path to input file')
    args = parser.parse_args()

    room_filter = None
    if args.filter_by_room == 'bronze':
        room_filter = 0
    elif args.filter_by_room == 'silver':
        room_filter = 1
    elif args.filter_by_room == 'gold':
        room_filter = 2
    elif args.filter_by_room == 'jade':
        room_filter = 3
    elif args.filter_by_room == 'throne':
        room_filter = 4
    else:
        raise RuntimeError(f'{args.filter_by_room}: An invalid room name.')
    assert room_filter is not None

    input_path: Optional[Path] = args.INPUT

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

    _parse(input_lines, room_filter, args.td_aware, args.curriculum)


if __name__ == '__main__':
    _main()
