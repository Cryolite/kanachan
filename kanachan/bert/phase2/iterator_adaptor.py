#!/usr/bin/env python3

from pathlib import Path
import torch
from kanachan.iterator_adaptor_base import IteratorAdaptorBase


class IteratorAdaptor(IteratorAdaptorBase):
    def __init__(self, path: Path, num_dimensions: int) -> None:
        super(IteratorAdaptor, self).__init__(path, num_dimensions)

    def __next__(self):
        sparse, numeric, positional, candidates, index, results = super(IteratorAdaptor, self).__next__()
        round_delta_of_score = results[1]
        alpha = 0.0011
        if alpha == 0.0:
            round_delta_of_score /= 10000.0
        else:
            import math
            score_magnitude = math.log1p(alpha * math.fabs(round_delta_of_score))
            round_delta_of_score = math.copysign(
                score_magnitude, round_delta_of_score)
            round_delta_of_score /= 10.0
        round_delta_of_score = torch.tensor(
            round_delta_of_score, device='cpu', dtype=torch.float32)
        return (sparse, numeric, positional, candidates, index, round_delta_of_score)
