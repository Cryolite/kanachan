from pathlib import Path
from typing import Tuple
import torch
from kanachan.training.iterator_adaptor_base import IteratorAdaptorBase


class IteratorAdaptor(IteratorAdaptorBase):
    def __init__(self, path: Path) -> None:
        super(IteratorAdaptor, self).__init__(path)

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sparse, numeric, positional, candidates, index, results = super(IteratorAdaptor, self).__next__()

        game_delta_of_grading_score = results[11] / 100.0

        return (sparse, numeric, positional, candidates, index, game_delta_of_grading_score)
