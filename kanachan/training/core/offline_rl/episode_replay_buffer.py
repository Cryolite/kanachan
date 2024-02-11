from pathlib import Path
from tqdm import tqdm
import torch
from torch import Tensor
from tensordict import TensorDict
from torchrl.data import ListStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from kanachan.constants import (
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ROUND_SUMMARY,
    RL_NUM_RESULTS,
)
from kanachan.training.common import get_distributed_environment
from kanachan.training.core.rl import RewardFunction
from kanachan.training.core.offline_rl.data_iterator import DataIterator


Annotation = tuple[
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


class EpisodeReplayBuffer:
    def __init__(
        self,
        *,
        training_data: Path,
        contiguous_training_data: bool,
        num_skip_samples: int,
        get_reward: RewardFunction,
        max_size: int,
        batch_size: int,
        drop_last: bool,
    ) -> None:
        if not training_data.exists():
            raise ValueError(training_data)
        if not training_data.is_file():
            raise ValueError(training_data)
        if max_size <= 0:
            raise ValueError(max_size)
        if batch_size <= 0:
            raise ValueError(batch_size)
        if batch_size > max_size:
            raise ValueError(batch_size)

        world_size, _, local_rank = get_distributed_environment()

        self.__data_iterator = DataIterator(
            path=training_data,
            num_skip_samples=num_skip_samples,
            local_rank=local_rank,
        )
        self.__contiguous = contiguous_training_data
        self.__get_reward = get_reward
        storage = ListStorage(max_size=max_size)
        sampler = SamplerWithoutReplacement(drop_last=drop_last)
        self.__replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=sampler,
            batch_size=(batch_size * world_size),
        )
        self.__batch_size = batch_size
        self.__max_size = max_size
        self.__size = 0
        self.__batch_buffer: list[TensorDict] = []
        self.__first_iteration = True

    def __iter__(self) -> "EpisodeReplayBuffer":
        return self

    def __next__(self) -> TensorDict:
        if len(self.__batch_buffer) >= 1:
            return self.__batch_buffer.pop(0)

        world_size, rank, local_rank = get_distributed_environment()

        progress: tqdm | None = None
        if self.__first_iteration:
            progress = tqdm(
                desc="Loading data to replay buffer...",
                total=self.__max_size,
                maxinterval=0.1,
                disable=(local_rank != 0),
                unit="samples",
                smoothing=0.0,
            )
            self.__first_iteration = False

        annotations: list[TensorDict] = []
        while True:
            try:
                annotation: Annotation = next(self.__data_iterator)
                assert len(annotation) == 13
            except StopIteration:
                assert len(annotations) == 0
                raise

            sparse = annotation[0]
            assert sparse.dim() == 1
            assert sparse.size(0) == MAX_NUM_ACTIVE_SPARSE_FEATURES
            assert sparse.dtype == torch.int32
            numeric = annotation[1]
            assert numeric.dim() == 1
            assert numeric.size(0) == NUM_NUMERIC_FEATURES
            assert numeric.dtype == torch.int32
            progression = annotation[2]
            assert progression.dim() == 1
            assert progression.size(0) == MAX_LENGTH_OF_PROGRESSION_FEATURES
            assert progression.dtype == torch.int32
            candidates = annotation[3]
            assert candidates.dim() == 1
            assert candidates.size(0) == MAX_NUM_ACTION_CANDIDATES
            assert candidates.dtype == torch.int32
            action = annotation[4]
            assert action.dim() == 0
            assert action.dtype == torch.int32
            next_sparse = annotation[5]
            assert next_sparse.dim() == 1
            assert next_sparse.size(0) == MAX_NUM_ACTIVE_SPARSE_FEATURES
            assert next_sparse.dtype == torch.int32
            next_numeric = annotation[6]
            assert next_numeric.dim() == 1
            assert next_numeric.size(0) == NUM_NUMERIC_FEATURES
            assert next_numeric.dtype == torch.int32
            next_progression = annotation[7]
            assert next_progression.dim() == 1
            assert (
                next_progression.size(0) == MAX_LENGTH_OF_PROGRESSION_FEATURES
            )
            assert next_progression.dtype == torch.int32
            next_candidates = annotation[8]
            assert next_candidates.dim() == 1
            assert next_candidates.size(0) == MAX_NUM_ACTION_CANDIDATES
            assert next_candidates.dtype == torch.int32
            round_summary = annotation[9]
            assert round_summary.dim() == 1
            assert round_summary.size(0) == MAX_NUM_ROUND_SUMMARY
            assert round_summary.dtype == torch.int32
            results = annotation[10]
            assert results.dim() == 1
            assert results.size(0) == RL_NUM_RESULTS
            assert results.dtype == torch.int32
            beginning_of_round = annotation[11]
            assert beginning_of_round.dim() == 0
            assert beginning_of_round.dtype == torch.bool
            done = annotation[12]
            assert done.dim() == 0
            assert done.dtype == torch.bool

            td = TensorDict(
                {
                    "sparse": sparse.unsqueeze(0),
                    "numeric": numeric.unsqueeze(0),
                    "progression": progression.unsqueeze(0),
                    "candidates": candidates.unsqueeze(0),
                    "action": action.unsqueeze(0),
                    "next": {
                        "sparse": next_sparse.unsqueeze(0),
                        "numeric": next_numeric.unsqueeze(0),
                        "progression": next_progression.unsqueeze(0),
                        "candidates": next_candidates.unsqueeze(0),
                        "round_summary": round_summary.unsqueeze(0),
                        "results": results.unsqueeze(0),
                        "end_of_round": beginning_of_round.unsqueeze(0),
                        "end_of_game": done.detach().clone().unsqueeze(0),
                        "done": done.unsqueeze(0),
                    },
                },
                batch_size=1,
                device=torch.device("cpu"),
            )
            annotations.append(td)

            if done.item():
                length = len(annotations)

                episode: TensorDict = torch.cat(annotations)  # type: ignore
                with torch.no_grad():
                    self.__get_reward(episode, self.__contiguous)
                if episode.get(("next", "reward"), None) is None:
                    errmsg = (
                        "`get_reward` did not set the"
                        " `('next', 'reward')` tensor."
                    )
                    raise RuntimeError(errmsg)
                reward: Tensor = episode["next", "reward"]
                if reward.dim() not in (1, 2):
                    errmsg = "An invalid shape of the `reward` tensor."
                    raise RuntimeError(errmsg)
                if reward.dim() == 2:
                    if reward.size(1) != 1:
                        errmsg = "An invalid shape of the `reward` tensor."
                        raise RuntimeError(errmsg)
                    reward.squeeze_(1)
                if reward.size(0) != length:
                    errmsg = "An invalid shape of the `reward` tensor."
                    raise RuntimeError(errmsg)
                if reward.dtype not in (
                    torch.float64,
                    torch.float32,
                    torch.float16,
                ):
                    errmsg = "An invalid `dtype` of the `reward` tensor."
                    raise RuntimeError(errmsg)

                if progress is not None:
                    if self.__size + length <= self.__max_size:
                        progress.update(length)
                    else:
                        progress.update(self.__max_size - self.__size)

                flag = False
                while True:
                    if self.__size + length > self.__max_size:
                        batch = self.__replay_buffer.sample().to_tensordict()
                        assert batch.size(0) == self.__batch_size * world_size
                        self.__size -= self.__batch_size * world_size
                        if world_size >= 2:
                            batch = batch[
                                self.__batch_size * rank : self.__batch_size
                                * (rank + 1)
                            ]
                        self.__batch_buffer.append(batch)
                        flag = True
                        continue
                    break
                assert self.__size + length <= self.__max_size
                self.__replay_buffer.extend(episode)
                self.__size += length
                annotations = []
                if flag:
                    if progress is not None:
                        progress.close()
                    assert len(self.__batch_buffer) >= 1
                    return self.__batch_buffer.pop(0)
