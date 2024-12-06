from typing import Callable, Any
import torch
from torch import Tensor, nn
from tensordict import TensorDictBase, TensorDict  # type: ignore
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
import kanachan.simulation
from kanachan.training.core.rl import RewardFunction


_RoundKey = tuple[int, int, int]
Progress = Callable[[], None]


@torch.no_grad()
def simulate(
    *,
    device: torch.device,
    dtype: torch.dtype,
    room: int,
    dongfengzhan: bool,
    grade: int,
    baseline_model: nn.Module,
    policy_model: nn.Module,
    num_simulations: int,
    simulation_batch_size: int,
    simulation_concurrency: int,
    progress: Progress,
    get_reward: RewardFunction,
) -> TensorDict:
    simulation_mode = 0
    simulation_mode |= 1  # non-duplicate mode
    if dongfengzhan:
        simulation_mode |= 2
    simulation_mode |= 4  # 1vs3 mode

    game_logs = kanachan.simulation.simulate(
        device,
        dtype,
        room,
        grade,
        baseline_model,
        grade,
        policy_model,
        simulation_mode,
        num_simulations,
        simulation_batch_size,
        simulation_concurrency,
        progress,
    )

    _episodes: list[TensorDict] = []
    for game_log in game_logs:
        game_result = game_log.get_result()
        round_results: dict[_RoundKey, dict[str, Any]] = {}
        for i in range(4):
            _round_results = game_result[i]["round_results"]
            for round_result in _round_results:
                chang: int = round_result["chang"]
                ju: int = round_result["ju"]
                benchang: int = round_result["benchang"]
                round_key = (chang, ju, benchang)
                delta_score: int = round_result["delta_score"]
                eor_score: int = round_result["score"]
                if round_key not in round_results:
                    round_results[round_key] = {}
                    round_results[round_key]["delta_scores"] = [0, 0, 0, 0]
                    round_results[round_key]["scores"] = [0, 0, 0, 0]
                round_results[round_key]["delta_scores"][i] = delta_score
                round_results[round_key]["scores"][i] = eor_score

        episode_meta_data = None
        for seat in range(4):
            episode_meta_data = game_log.get_episode(seat)
            if episode_meta_data["proposed"]:
                break
        assert episode_meta_data is not None

        game_scores = episode_meta_data["scores"]
        assert isinstance(game_scores, list)
        assert len(game_scores) == 4
        assert all(isinstance(game_score, int) for game_score in game_scores)

        episode = episode_meta_data["episode"]
        assert isinstance(episode, TensorDictBase)
        episode = episode.to(torch.device("cpu")).to_tensordict()
        assert isinstance(episode, TensorDict)

        length = episode.size(0)
        assert isinstance(length, int)

        sparse = episode["sparse"]
        assert isinstance(sparse, Tensor)
        assert sparse.device == torch.device("cpu")
        assert sparse.dtype == torch.int32
        assert sparse.dim() == 2
        assert sparse.size(0) == length
        assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
        assert torch.all(sparse >= 0).item()
        assert torch.all(sparse <= NUM_TYPES_OF_SPARSE_FEATURES).item()
        numeric = episode["numeric"]
        assert isinstance(numeric, Tensor)
        assert numeric.device == torch.device("cpu")
        assert numeric.dtype == torch.int32
        assert numeric.dim() == 2
        assert numeric.size(0) == length
        assert numeric.size(1) == NUM_NUMERIC_FEATURES
        progression = episode["progression"]
        assert isinstance(progression, Tensor)
        assert progression.device == torch.device("cpu")
        assert progression.dtype == torch.int32
        assert progression.dim() == 2
        assert progression.size(0) == length
        assert progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
        assert torch.all(progression >= 0).item()
        assert torch.all(
            progression <= NUM_TYPES_OF_PROGRESSION_FEATURES
        ).item()
        candidates = episode["candidates"]
        assert isinstance(candidates, Tensor)
        assert candidates.device == torch.device("cpu")
        assert candidates.dtype == torch.int32
        assert candidates.dim() == 2
        assert candidates.size(0) == length
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
        assert torch.all(candidates >= 0).item()
        assert torch.all(candidates <= NUM_TYPES_OF_ACTIONS).item()
        action_index = episode["action"]
        assert isinstance(action_index, Tensor)
        assert action_index.device == torch.device("cpu")
        assert action_index.dtype == torch.int32
        assert action_index.dim() == 1
        assert action_index.size(0) == length
        assert torch.all(action_index >= 0).item()
        assert torch.all(action_index < MAX_NUM_ACTION_CANDIDATES).item()
        log_prob = episode.get("sample_log_prob", None)  # type: ignore
        if log_prob is not None:
            assert log_prob.device == torch.device("cpu")
            assert log_prob.dtype == torch.float64
            assert log_prob.dim() == 1
            assert log_prob.size(0) == length

        episode["next", "sparse"] = sparse.roll(-1, 0).detach().clone()
        episode["next", "sparse"][-1] = torch.full_like(
            sparse[-1], NUM_TYPES_OF_SPARSE_FEATURES
        )
        episode["next", "numeric"] = numeric.roll(-1, 0).detach().clone()
        episode["next", "numeric"][-1] = torch.full_like(numeric[-1], 0)
        episode["next", "progression"] = (
            progression.roll(-1, 0).detach().clone()
        )
        episode["next", "progression"][-1] = torch.full_like(
            progression[-1], NUM_TYPES_OF_PROGRESSION_FEATURES
        )
        episode["next", "candidates"] = candidates.roll(-1, 0).detach().clone()
        episode["next", "candidates"][-1] = torch.full_like(
            candidates[-1], NUM_TYPES_OF_ACTIONS
        )

        # TODO
        episode["next", "round_summary"] = torch.full(
            (length, MAX_NUM_ROUND_SUMMARY),
            NUM_TYPES_OF_ROUND_SUMMARY,
            device=torch.device("cpu"),
            dtype=torch.int32,
        )

        episode["next", "results"] = torch.full(
            (length, NUM_RESULTS),
            0,
            device=torch.device("cpu"),
            dtype=torch.int32,
        )
        episode["next", "end_of_round"] = torch.full(
            (length,), False, device=torch.device("cpu"), dtype=torch.bool
        )
        episode["next", "end_of_game"] = torch.full(
            (length,), False, device=torch.device("cpu"), dtype=torch.bool
        )
        for t in range(length):
            chang = int(sparse[t, 7].item()) - 75
            ju = int(sparse[t, 8].item()) - 78
            benchang = int(numeric[t, 0].item())
            round_key = (chang, ju, benchang)

            next_round_key: _RoundKey | None = None
            if t + 1 < length:
                next_chang = int(sparse[t + 1, 7].item()) - 75
                next_ju = int(sparse[t + 1, 8].item()) - 78
                next_benchang = int(numeric[t + 1, 0].item())
                next_round_key = (next_chang, next_ju, next_benchang)

            if round_key != next_round_key:
                episode["next", "results"][t][0] = round_results[round_key][
                    "delta_scores"
                ][0]
                episode["next", "results"][t][1] = round_results[round_key][
                    "delta_scores"
                ][1]
                episode["next", "results"][t][2] = round_results[round_key][
                    "delta_scores"
                ][2]
                episode["next", "results"][t][3] = round_results[round_key][
                    "delta_scores"
                ][3]
                # TODO
                episode["next", "results"][t][4] = 0
                # TODO
                episode["next", "results"][t][5] = 0
                episode["next", "results"][t][6] = round_results[round_key][
                    "scores"
                ][0]
                episode["next", "results"][t][7] = round_results[round_key][
                    "scores"
                ][1]
                episode["next", "results"][t][8] = round_results[round_key][
                    "scores"
                ][2]
                episode["next", "results"][t][9] = round_results[round_key][
                    "scores"
                ][3]
                episode["next", "results"][t][10] = game_scores[0]
                episode["next", "results"][t][11] = game_scores[1]
                episode["next", "results"][t][12] = game_scores[2]
                episode["next", "results"][t][13] = game_scores[3]
                episode["next", "end_of_round"][t] = True
        episode["next", "end_of_game"][-1] = True
        episode["next", "done"] = (
            episode["next", "end_of_game"].detach().clone()
        )

        get_reward(episode, True)
        if episode.get(("next", "reward"), None) is None:  # type: ignore
            errmsg = (
                "`get_reward` did not set the `('next', 'reward')` tensor."
            )
            raise RuntimeError(errmsg)
        reward: Tensor = episode["next", "reward"]
        assert isinstance(reward, Tensor)
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
        if reward.dtype not in (torch.float64, torch.float32, torch.float16):
            errmsg = "An invalid `dtype` for the `reward` tensor."
            raise RuntimeError(errmsg)
        episode["next", "reward"] = reward.to(dtype=dtype)

        _episodes.append(episode.detach().clone().cpu())

    episodes: TensorDict = torch.cat(_episodes).to_tensordict()  # type: ignore
    assert isinstance(episodes, TensorDict)

    return episodes
