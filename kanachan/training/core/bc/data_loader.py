from pathlib import Path
from torch import Tensor
import torch.utils.data
from tensordict import TensorDict
from kanachan.constants import (
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ROUND_SUMMARY,
    NUM_RESULTS,
)
from kanachan.training.common import get_distributed_environment
from kanachan.training.core.bc.data_iterator import DataIterator
from kanachan.training.core.dataset import Dataset


_Batch = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class DataLoader:
    def __init__(
        self,
        *,
        path: Path,
        num_skip_samples: int,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        dataset = Dataset(
            path=path,
            iterator_class=DataIterator,
            num_skip_samples=num_skip_samples,
        )
        world_size, _, _ = get_distributed_environment()
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=(batch_size * world_size),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        self.__iterator = iter(data_loader)

    def __iter__(self) -> "DataLoader":
        return self

    def _split_batch_in_case_of_multiprocess(self, batch: _Batch) -> _Batch:
        world_size, rank, _ = get_distributed_environment()
        assert batch[0].size(0) % world_size == 0
        batch_size = batch[0].size(0) // world_size
        if world_size >= 2:
            first = batch_size * rank
            last = batch_size * (rank + 1)
            batch = tuple(x[first:last] for x in batch)  # type: ignore

        return batch

    def __next__(self) -> TensorDict:
        batch: _Batch = next(self.__iterator)
        batch = self._split_batch_in_case_of_multiprocess(batch)

        assert batch[0].dim() == 2
        batch_size = batch[0].size(0)
        assert batch[0].size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
        assert batch[1].dim() == 2
        assert batch[1].size(0) == batch_size
        assert batch[1].size(1) == NUM_NUMERIC_FEATURES
        assert batch[2].dim() == 2
        assert batch[2].size(0) == batch_size
        assert batch[2].size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
        assert batch[3].dim() == 2
        assert batch[3].size(0) == batch_size
        assert batch[3].size(1) == MAX_NUM_ACTION_CANDIDATES
        assert batch[4].dim() == 1
        assert batch[4].size(0) == batch_size
        assert batch[5].dim() == 2
        assert batch[5].size(0) == batch_size
        assert batch[5].size(1) == MAX_NUM_ROUND_SUMMARY
        assert batch[6].dim() == 2
        assert batch[6].size(0) == batch_size
        assert batch[6].size(1) == NUM_RESULTS

        source = {
            "sparse": batch[0],
            "numeric": batch[1],
            "progression": batch[2],
            "candidates": batch[3],
            "action": batch[4],
            "round_summary": batch[5],
            "results": batch[6],
        }
        data = TensorDict(
            source=source, batch_size=batch_size, device=torch.device("cpu")
        )

        return data
