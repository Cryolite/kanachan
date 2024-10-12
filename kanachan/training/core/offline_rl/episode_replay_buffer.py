from pathlib import Path
import random
from typing import Any
from tqdm import tqdm
import torch
from torch import Tensor
import torch.utils.data
from torch.distributed import broadcast
from tensordict import TensorDict  # type: ignore
from kanachan.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES,
    NUM_TYPES_OF_ROUND_SUMMARY,
    MAX_NUM_ROUND_SUMMARY,
    NUM_RESULTS,
)
from kanachan.training.common import get_distributed_environment
from kanachan.training.core.rl import RewardFunction
from kanachan.training.core.offline_rl.dataset import Dataset


_INTERNAL_BATCH_SIZE = 1024
_MAIN_RANK = 0


_BUFFER_ELEMENT = tuple[
    Tensor,  # sparse
    Tensor,  # numeric
    Tensor,  # progression
    Tensor,  # candidates
    Tensor,  # action
    Tensor,  # next_sparse
    Tensor,  # next_numeric
    Tensor,  # next_progression
    Tensor,  # next_candidates
    Tensor,  # round_summary
    Tensor,  # results
    Tensor,  # end_of_round
    Tensor,  # end_of_game
    Tensor,  # done
    Tensor,  # reward
]


class EpisodeReplayBuffer:
    def __init__(
        self,
        *,
        training_data: Path,
        contiguous_training_data: bool,
        num_skip_samples: int,
        rewrite_rooms: int | None,
        rewrite_grades: int | None,
        get_reward: RewardFunction,
        dtype: torch.dtype,
        max_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        if not training_data.exists():
            raise RuntimeError(f"{training_data}: Does not exist.")
        if not training_data.is_file():
            raise RuntimeError(f"{training_data}: Not a file.")
        if num_skip_samples < 0:
            raise ValueError(num_skip_samples)
        if rewrite_rooms is not None:
            if rewrite_rooms < 0 or rewrite_rooms > 4:
                raise ValueError(rewrite_rooms)
        if rewrite_grades is not None:
            if rewrite_grades < 0 or rewrite_grades > 15:
                raise ValueError(rewrite_grades)
        if max_size <= 0:
            raise ValueError(max_size)
        if batch_size <= 0:
            raise ValueError(batch_size)
        if batch_size > max_size:
            raise ValueError(f"{batch_size} > {max_size}")
        if num_workers < 0:
            raise ValueError(num_workers)

        dataset = Dataset(
            path=training_data,
            num_skip_samples=num_skip_samples,
            rewrite_rooms=rewrite_rooms,
            rewrite_grades=rewrite_grades,
        )
        self.__data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=_INTERNAL_BATCH_SIZE,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.__data_iter = iter(self.__data_loader)
        self.__contiguous = contiguous_training_data
        self.__sparse: Tensor = torch.empty(
            0, MAX_NUM_ACTIVE_SPARSE_FEATURES, device="cpu", dtype=torch.int32
        )
        self.__numeric: Tensor = torch.empty(
            0, NUM_NUMERIC_FEATURES, device="cpu", dtype=torch.int32
        )
        self.__progression: Tensor = torch.empty(
            0,
            MAX_LENGTH_OF_PROGRESSION_FEATURES,
            device="cpu",
            dtype=torch.int32,
        )
        self.__candidates: Tensor = torch.empty(
            0, MAX_NUM_ACTION_CANDIDATES, device="cpu", dtype=torch.int32
        )
        self.__action: Tensor = torch.empty(0, device="cpu", dtype=torch.int32)
        self.__next_sparse: Tensor = torch.empty(
            0, MAX_NUM_ACTIVE_SPARSE_FEATURES, device="cpu", dtype=torch.int32
        )
        self.__next_numeric: Tensor = torch.empty(
            0, NUM_NUMERIC_FEATURES, device="cpu", dtype=torch.int32
        )
        self.__next_progression: Tensor = torch.empty(
            0,
            MAX_LENGTH_OF_PROGRESSION_FEATURES,
            device="cpu",
            dtype=torch.int32,
        )
        self.__next_candidates: Tensor = torch.empty(
            0, MAX_NUM_ACTION_CANDIDATES, device="cpu", dtype=torch.int32
        )
        self.__round_summary: Tensor = torch.empty(
            0, MAX_NUM_ROUND_SUMMARY, device="cpu", dtype=torch.int32
        )
        self.__results: Tensor = torch.empty(
            0, NUM_RESULTS, device="cpu", dtype=torch.int32
        )
        self.__end_of_round: Tensor = torch.empty(
            0, device="cpu", dtype=torch.bool
        )
        self.__done: Tensor = torch.empty(0, device="cpu", dtype=torch.bool)
        self.__get_reward = get_reward
        self.__dtype = dtype
        self.__replay_buffer: list[_BUFFER_ELEMENT] = []
        self.__batch_size = batch_size
        self.__max_size = max_size
        self.__first_iteration = True

    def __iter__(self) -> "EpisodeReplayBuffer":
        return self

    def _append(self, episode: TensorDict) -> None:
        for i in range(episode.batch_size[0]):
            td = episode[i].to_tensordict()  # type: ignore
            sparse = td["sparse"]
            numeric = td["numeric"]
            progression = td["progression"]
            candidates = td["candidates"]
            action = td["action"]
            next_sparse = td["next", "sparse"]
            next_numeric = td["next", "numeric"]
            next_progression = td["next", "progression"]
            next_candidates = td["next", "candidates"]
            round_summary = td["next", "round_summary"]
            results = td["next", "results"]
            end_of_round = td["next", "end_of_round"]
            end_of_game = td["next", "end_of_game"]
            done = td["next", "done"]
            reward = td["next", "reward"]
            self.__replay_buffer.append(
                (
                    sparse,
                    numeric,
                    progression,
                    candidates,
                    action,
                    next_sparse,
                    next_numeric,
                    next_progression,
                    next_candidates,
                    round_summary,
                    results,
                    end_of_round,
                    end_of_game,
                    done,
                    reward,
                )
            )

    def _sample(self) -> _BUFFER_ELEMENT:
        if len(self.__replay_buffer) == 0:
            errmsg = "The replay buffer is empty."
            raise RuntimeError(errmsg)
        idx = random.randrange(len(self.__replay_buffer))
        return self.__replay_buffer.pop(idx)

    def _sample_batch(self) -> _BUFFER_ELEMENT:
        world_size, _, _ = get_distributed_environment()
        if len(self.__replay_buffer) < self.__batch_size * world_size:
            errmsg = "The replay buffer is too small."
            raise RuntimeError(errmsg)

        batch: list[list[Tensor]] = [[] for _ in range(15)]
        for _ in range(self.__batch_size * world_size):
            t = self._sample()
            for i in range(15):
                batch[i].append(t[i])

        return (
            torch.stack(batch[0]),
            torch.stack(batch[1]),
            torch.stack(batch[2]),
            torch.stack(batch[3]),
            torch.stack(batch[4]),
            torch.stack(batch[5]),
            torch.stack(batch[6]),
            torch.stack(batch[7]),
            torch.stack(batch[8]),
            torch.stack(batch[9]),
            torch.stack(batch[10]),
            torch.stack(batch[11]),
            torch.stack(batch[12]),
            torch.stack(batch[13]),
            torch.stack(batch[14]),
        )

    def _broadcast(self) -> TensorDict:
        world_size, rank, _ = get_distributed_environment()

        if rank == _MAIN_RANK:
            (
                sparse,
                numeric,
                progression,
                candidates,
                action,
                next_sparse,
                next_numeric,
                next_progression,
                next_candidates,
                round_summary,
                results,
                end_of_round,
                end_of_game,
                done,
                reward,
            ) = self._sample_batch()
        else:
            sparse = torch.empty(
                self.__batch_size * world_size,
                MAX_NUM_ACTIVE_SPARSE_FEATURES,
                device="cpu",
                dtype=torch.int32,
            )
            numeric = torch.empty(
                self.__batch_size * world_size,
                NUM_NUMERIC_FEATURES,
                device="cpu",
                dtype=torch.int32,
            )
            progression = torch.empty(
                self.__batch_size * world_size,
                MAX_LENGTH_OF_PROGRESSION_FEATURES,
                device="cpu",
                dtype=torch.int32,
            )
            candidates = torch.empty(
                self.__batch_size * world_size,
                MAX_NUM_ACTION_CANDIDATES,
                device="cpu",
                dtype=torch.int32,
            )
            action = torch.empty(
                self.__batch_size * world_size,
                device="cpu",
                dtype=torch.int32,
            )
            next_sparse = torch.empty(
                self.__batch_size * world_size,
                MAX_NUM_ACTIVE_SPARSE_FEATURES,
                device="cpu",
                dtype=torch.int32,
            )
            next_numeric = torch.empty(
                self.__batch_size * world_size,
                NUM_NUMERIC_FEATURES,
                device="cpu",
                dtype=torch.int32,
            )
            next_progression = torch.empty(
                self.__batch_size * world_size,
                MAX_LENGTH_OF_PROGRESSION_FEATURES,
                device="cpu",
                dtype=torch.int32,
            )
            next_candidates = torch.empty(
                self.__batch_size * world_size,
                MAX_NUM_ACTION_CANDIDATES,
                device="cpu",
                dtype=torch.int32,
            )
            round_summary = torch.empty(
                self.__batch_size * world_size,
                MAX_NUM_ROUND_SUMMARY,
                device="cpu",
                dtype=torch.int32,
            )
            results = torch.empty(
                self.__batch_size * world_size,
                NUM_RESULTS,
                device="cpu",
                dtype=torch.int32,
            )
            end_of_round = torch.empty(
                self.__batch_size * world_size,
                device="cpu",
                dtype=torch.bool,
            )
            end_of_game = torch.empty(
                self.__batch_size * world_size,
                device="cpu",
                dtype=torch.bool,
            )
            done = torch.empty(
                self.__batch_size * world_size,
                device="cpu",
                dtype=torch.bool,
            )
            reward = torch.empty(
                self.__batch_size * world_size,
                device="cpu",
                dtype=self.__dtype,
            )

        if world_size >= 2:
            tensors = [
                sparse,
                numeric,
                progression,
                candidates,
                action,
                next_sparse,
                next_numeric,
                next_progression,
                next_candidates,
                round_summary,
                results,
                end_of_round,
                end_of_game,
                done,
                reward,
            ]
            for i, t in enumerate(tensors):
                t = t.to(device="cuda")
                broadcast(t, src=_MAIN_RANK)
                t = t.to(device="cpu")
                first = self.__batch_size * rank
                last = self.__batch_size * (rank + 1)
                tensors[i] = t[first:last]
            (
                sparse,
                numeric,
                progression,
                candidates,
                action,
                next_sparse,
                next_numeric,
                next_progression,
                next_candidates,
                round_summary,
                results,
                end_of_round,
                end_of_game,
                done,
                reward,
            ) = tensors

        src = {
            "sparse": sparse,
            "numeric": numeric,
            "progression": progression,
            "candidates": candidates,
            "action": action,
            "next": {
                "sparse": next_sparse,
                "numeric": next_numeric,
                "progression": next_progression,
                "candidates": next_candidates,
                "round_summary": round_summary,
                "results": results,
                "end_of_round": end_of_round,
                "end_of_game": end_of_game,
                "done": done,
                "reward": reward,
            },
        }
        return TensorDict(
            src,
            batch_size=self.__batch_size,
            device=torch.device("cpu"),
        )

    def __next__(self) -> TensorDict:
        world_size, rank, local_rank = get_distributed_environment()

        if rank != _MAIN_RANK:
            return self._broadcast()

        if (
            len(self.__replay_buffer)
            >= self.__max_size + self.__batch_size * world_size
        ):
            return self._broadcast()

        progress: tqdm | None = None
        if self.__first_iteration:
            progress = tqdm(
                desc="Loading data to replay buffer...",
                total=self.__max_size,
                maxinterval=0.1,
                disable=(local_rank != 0),
                unit=" samples",
                smoothing=0.0,
            )
            self.__first_iteration = False

        while True:
            try:
                data = next(self.__data_iter)
            except StopIteration:
                if progress is not None:
                    progress.close()
                if len(self.__replay_buffer) >= self.__batch_size * world_size:
                    return self._broadcast()
                raise

            assert isinstance(data, list)
            assert len(data) == 13

            sparse: Tensor = data[0]
            assert isinstance(sparse, Tensor)
            assert sparse.device == torch.device("cpu")
            assert sparse.dtype == torch.int32
            assert sparse.dim() == 2
            assert sparse.size(0) == _INTERNAL_BATCH_SIZE
            assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
            assert torch.all(sparse >= 0).item()
            assert torch.all(sparse <= NUM_TYPES_OF_SPARSE_FEATURES).item()
            self.__sparse = torch.cat((self.__sparse, sparse), dim=0)

            numeric: Tensor = data[1]
            assert isinstance(numeric, Tensor)
            assert numeric.device == torch.device("cpu")
            assert numeric.dtype == torch.int32
            assert numeric.dim() == 2
            assert numeric.size(0) == _INTERNAL_BATCH_SIZE
            assert numeric.size(1) == NUM_NUMERIC_FEATURES
            self.__numeric = torch.cat((self.__numeric, numeric), dim=0)

            progression: Tensor = data[2]
            assert isinstance(progression, Tensor)
            assert progression.device == torch.device("cpu")
            assert progression.dtype == torch.int32
            assert progression.dim() == 2
            assert progression.size(0) == _INTERNAL_BATCH_SIZE
            assert progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
            assert torch.all(progression >= 0).item()
            assert torch.all(
                progression <= NUM_TYPES_OF_PROGRESSION_FEATURES
            ).item()
            self.__progression = torch.cat(
                (self.__progression, progression), dim=0
            )

            candidates: Tensor = data[3]
            assert isinstance(candidates, Tensor)
            assert candidates.dtype == torch.int32
            assert candidates.device == torch.device("cpu")
            assert candidates.dim() == 2
            assert candidates.size(0) == _INTERNAL_BATCH_SIZE
            assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
            assert torch.all(candidates >= 0).item()
            assert torch.all(candidates <= NUM_TYPES_OF_ACTIONS).item()
            self.__candidates = torch.cat(
                (self.__candidates, candidates), dim=0
            )

            action: Tensor = data[4]
            assert isinstance(action, Tensor)
            assert action.dtype == torch.int32
            assert action.dim() == 1
            assert action.size(0) == _INTERNAL_BATCH_SIZE
            assert torch.all(action >= 0).item()
            assert torch.all(action < MAX_NUM_ACTION_CANDIDATES).item()
            self.__action = torch.cat((self.__action, action), dim=0)

            next_sparse: Tensor = data[5]
            assert isinstance(next_sparse, Tensor)
            assert next_sparse.device == torch.device("cpu")
            assert next_sparse.dtype == torch.int32
            assert next_sparse.dim() == 2
            assert next_sparse.size(0) == _INTERNAL_BATCH_SIZE
            assert next_sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
            assert torch.all(next_sparse >= 0).item()
            assert torch.all(
                next_sparse <= NUM_TYPES_OF_SPARSE_FEATURES
            ).item()
            self.__next_sparse = torch.cat(
                (self.__next_sparse, next_sparse), dim=0
            )

            next_numeric: Tensor = data[6]
            assert isinstance(next_numeric, Tensor)
            assert next_numeric.device == torch.device("cpu")
            assert next_numeric.dtype == torch.int32
            assert next_numeric.dim() == 2
            assert next_numeric.size(0) == _INTERNAL_BATCH_SIZE
            assert next_numeric.size(1) == NUM_NUMERIC_FEATURES
            self.__next_numeric = torch.cat(
                (self.__next_numeric, next_numeric), dim=0
            )

            next_progression: Tensor = data[7]
            assert isinstance(next_progression, Tensor)
            assert next_progression.device == torch.device("cpu")
            assert next_progression.dtype == torch.int32
            assert next_progression.dim() == 2
            assert next_progression.size(0) == _INTERNAL_BATCH_SIZE
            assert (
                next_progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
            )
            assert torch.all(next_progression >= 0).item()
            assert torch.all(
                next_progression <= NUM_TYPES_OF_PROGRESSION_FEATURES
            ).item()
            self.__next_progression = torch.cat(
                (self.__next_progression, next_progression), dim=0
            )

            next_candidates: Tensor = data[8]
            assert isinstance(next_candidates, Tensor)
            assert next_candidates.device == torch.device("cpu")
            assert next_candidates.dtype == torch.int32
            assert next_candidates.dim() == 2
            assert next_candidates.size(0) == _INTERNAL_BATCH_SIZE
            assert next_candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
            assert torch.all(next_candidates >= 0).item()
            assert torch.all(next_candidates <= NUM_TYPES_OF_ACTIONS).item()
            self.__next_candidates = torch.cat(
                (self.__next_candidates, next_candidates), dim=0
            )

            round_summary: Tensor = data[9]
            assert isinstance(round_summary, Tensor)
            assert round_summary.device == torch.device("cpu")
            assert round_summary.dtype == torch.int32
            assert round_summary.dim() == 2
            assert round_summary.size(0) == _INTERNAL_BATCH_SIZE
            assert round_summary.size(1) == MAX_NUM_ROUND_SUMMARY
            assert torch.all(round_summary >= 0).item()
            assert torch.all(
                round_summary <= NUM_TYPES_OF_ROUND_SUMMARY
            ).item()
            self.__round_summary = torch.cat(
                (self.__round_summary, round_summary), dim=0
            )

            results: Tensor = data[10]
            assert isinstance(results, Tensor)
            assert results.device == torch.device("cpu")
            assert results.dtype == torch.int32
            assert results.dim() == 2
            assert results.size(0) == _INTERNAL_BATCH_SIZE
            assert results.size(1) == NUM_RESULTS
            self.__results = torch.cat((self.__results, results), dim=0)

            end_of_round: Tensor = data[11]
            assert isinstance(end_of_round, Tensor)
            assert end_of_round.device == torch.device("cpu")
            assert end_of_round.dtype == torch.bool
            assert end_of_round.dim() == 1
            assert end_of_round.size(0) == _INTERNAL_BATCH_SIZE
            self.__end_of_round = torch.cat(
                (self.__end_of_round, end_of_round), dim=0
            )

            done: Tensor = data[12]
            assert isinstance(done, Tensor)
            assert done.device == torch.device("cpu")
            assert done.dtype == torch.bool
            assert done.dim() == 1
            assert done.size(0) == _INTERNAL_BATCH_SIZE
            self.__done = torch.cat((self.__done, done), dim=0)

            length: int | None = None
            for i, _ in enumerate(self.__done):
                if self.__done[i]:
                    length = i + 1
            if length is None:
                continue

            source: dict[str, Any] = {
                "sparse": self.__sparse[:length],
                "numeric": self.__numeric[:length],
                "progression": self.__progression[:length],
                "candidates": self.__candidates[:length],
                "action": self.__action[:length],
                "next": {
                    "sparse": self.__next_sparse[:length],
                    "numeric": self.__next_numeric[:length],
                    "progression": self.__next_progression[:length],
                    "candidates": self.__next_candidates[:length],
                    "round_summary": self.__round_summary[:length],
                    "results": self.__results[:length],
                    "end_of_game": self.__done[:length],
                    "end_of_round": self.__end_of_round[:length],
                    "done": self.__done[:length].clone(),
                },
            }
            episode = TensorDict(
                source,
                batch_size=length,
                device=torch.device("cpu"),
            )

            with torch.no_grad():
                self.__get_reward(episode, self.__contiguous)
            if episode.get(("next", "reward"), None) is None:  # type: ignore
                errmsg = (
                    "`get_reward` did not set the "
                    '`("next", "reward")` tensor.'
                )
                raise RuntimeError(errmsg)
            reward: Tensor = episode["next", "reward"]
            if not isinstance(reward, Tensor):
                errmsg = "The `reward` is not a tensor."
                raise RuntimeError(errmsg)
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
            episode["next", "reward"] = (
                reward.to(self.__dtype).detach().clone()
            )

            if progress is not None:
                if len(self.__replay_buffer) + length <= self.__max_size:
                    progress.update(length)
                elif len(self.__replay_buffer) < self.__max_size:
                    progress.update(
                        self.__max_size - len(self.__replay_buffer)
                    )

            self._append(episode)
            self.__sparse = self.__sparse[length:]
            self.__numeric = self.__numeric[length:]
            self.__progression = self.__progression[length:]
            self.__candidates = self.__candidates[length:]
            self.__action = self.__action[length:]
            self.__next_sparse = self.__next_sparse[length:]
            self.__next_numeric = self.__next_numeric[length:]
            self.__next_progression = self.__next_progression[length:]
            self.__next_candidates = self.__next_candidates[length:]
            self.__round_summary = self.__round_summary[length:]
            self.__results = self.__results[length:]
            self.__end_of_round = self.__end_of_round[length:]
            self.__done = self.__done[length:]

            if (
                len(self.__replay_buffer)
                >= self.__max_size + self.__batch_size * world_size
            ):
                if progress is not None:
                    progress.close()
                return self._broadcast()

        raise RuntimeError("Never reach here.")

    def __len__(self) -> int:
        return len(self.__replay_buffer)
