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
    RL_NUM_RESULTS,
)
from kanachan.training.common import get_distributed_environment
from kanachan.training.core.offline_rl.data_iterator import DataIterator
from kanachan.training.core.dataset import Dataset
from kanachan.training.core.rl.reward_function import RewardFunction


_Batch = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


class DataLoader:
    def __init__(
        self,
        *,
        path: Path,
        num_skip_samples: int,
        get_reward: RewardFunction,
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
        self.__get_reward = get_reward

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
        assert batch[5].size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
        assert batch[6].dim() == 2
        assert batch[6].size(0) == batch_size
        assert batch[6].size(1) == NUM_NUMERIC_FEATURES
        assert batch[7].dim() == 2
        assert batch[7].size(0) == batch_size
        assert batch[7].size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
        assert batch[8].dim() == 2
        assert batch[8].size(0) == batch_size
        assert batch[8].size(1) == MAX_NUM_ACTION_CANDIDATES
        assert batch[9].dim() == 2
        assert batch[9].size(0) == batch_size
        assert batch[9].size(1) == MAX_NUM_ROUND_SUMMARY
        assert batch[10].dim() == 2
        assert batch[10].size(0) == batch_size
        assert batch[10].size(1) == RL_NUM_RESULTS
        assert batch[11].dim() == 1
        assert batch[11].size(0) == batch_size
        assert batch[12].dim() == 1
        assert batch[12].size(0) == batch_size

        source = {
            "sparse": batch[0],
            "numeric": batch[1],
            "progression": batch[2],
            "candidates": batch[3],
            "action": batch[4],
            "next": {
                "sparse": batch[5],
                "numeric": batch[6],
                "progression": batch[7],
                "candidates": batch[8],
                "round_summary": batch[9],
                "results": batch[10],
                "end_of_round": batch[11],
                "end_of_game": batch[12].detach().clone(),
                "done": batch[12],
            },
        }
        data = TensorDict(
            source=source, batch_size=batch_size, device=torch.device("cpu")
        )

        with torch.no_grad():
            self.__get_reward(data, False)
        if data.get(("next", "reward"), None) is None:
            errmsg = "`get_reward` did not set the `reward` tensor."
            raise RuntimeError(errmsg)
        reward: Tensor = data["next", "reward"]
        if reward.dim() not in (1, 2):
            errmsg = "An invalid shape of the `reward` tensor."
            raise RuntimeError(errmsg)
        if reward.dim() == 2:
            if reward.size(1) != 1:
                errmsg = "An invalid shape of the `reward` tensor."
                raise RuntimeError(errmsg)
            reward.squeeze_(1)
        if reward.size(0) != batch_size:
            errmsg = "An invalid shape of the `reward` tensor."
            raise RuntimeError(errmsg)
        if reward.dtype not in (torch.float64, torch.float32, torch.float16):
            errmsg = "An invalid `dtype` for the `reward` tensor."
            raise RuntimeError(errmsg)

        return data
