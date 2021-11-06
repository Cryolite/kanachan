#!/usr/bin/env python3

import pathlib
import torch
from kanachan.constants import (SEAT_INDEX, SEAT_OFFSET,)
from kanachan.iterator_adaptor_base import IteratorAdaptorBase


class IteratorAdaptor(IteratorAdaptorBase):
    def __init__(self, path: pathlib.Path, num_dimensions: int) -> None:
        super(IteratorAdaptor, self).__init__(path, num_dimensions)

    def __next__(self):
        sparse, numeric, positional, candidates, index, results = super(IteratorAdaptor, self).__next__()
        seat = sparse[SEAT_INDEX] - SEAT_OFFSET
        assert(0 <= seat and seat < 4)
        round_delta_of_score = results[seat]
        round_delta_of_score /= 10000.0
        round_delta_of_score = torch.tensor(
            round_delta_of_score, device='cpu', dtype=torch.float32)
        return (sparse, numeric, positional, candidates, index, round_delta_of_score)
