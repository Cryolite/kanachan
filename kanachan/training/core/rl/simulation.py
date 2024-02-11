from typing import Callable
import torch
from torch import Tensor, nn
from tensordict import TensorDictBase, TensorDict
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
    RL_NUM_RESULTS,
)
import kanachan.simulation
from kanachan.training.core.rl import RewardFunction


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
    delete_baseline_keys: list[str],
    policy_model: nn.Module,
    delete_proposed_keys: list[str],
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
        delete_baseline_keys,
        grade,
        policy_model,
        delete_proposed_keys,
        simulation_mode,
        num_simulations,
        simulation_batch_size,
        simulation_concurrency,
        progress,
    )

    _episodes: list[TensorDict] = []
    for game_log in game_logs:
        game_result = game_log.get_result()
        round_results = {}
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

        episode = episode_meta_data["episode"]
        assert isinstance(episode, TensorDictBase)
        episode = episode.to(torch.device("cpu")).to_tensordict()

        length = int(episode.size(0))  # type: ignore

        sparse: Tensor = episode["sparse"]
        assert sparse.dim() == 2
        assert sparse.size(0) == length
        assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
        numeric: Tensor = episode["numeric"]
        assert numeric.dim() == 2
        assert numeric.size(0) == length
        assert numeric.size(1) == NUM_NUMERIC_FEATURES
        progression: Tensor = episode["progression"]
        assert progression.dim() == 2
        assert progression.size(0) == length
        assert progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
        candidates: Tensor = episode["candidates"]
        assert candidates.dim() == 2
        assert candidates.size(0) == length
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
        action_index: Tensor = episode["action"]
        assert action_index.dim() == 1
        assert action_index.size(0) == length
        log_prob: Tensor | None = episode.get("sample_log_prob", None)
        if log_prob is not None:
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
            (length, RL_NUM_RESULTS),
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

            next_round_key = (None, None, None)
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
                episode["next", "results"][t][4] = round_results[round_key][
                    "scores"
                ][0]
                episode["next", "results"][t][5] = round_results[round_key][
                    "scores"
                ][1]
                episode["next", "results"][t][6] = round_results[round_key][
                    "scores"
                ][2]
                episode["next", "results"][t][7] = round_results[round_key][
                    "scores"
                ][3]
                episode["next", "end_of_round"][t] = True
        episode["next", "end_of_game"][-1] = True
        episode["next", "done"] = (
            episode["next", "end_of_game"].detach().clone()
        )

        get_reward(episode, True)
        if episode.get(("next", "reward"), None) is None:
            errmsg = (
                "`get_reward` did not set the `('next', 'reward')` tensor."
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
        if reward.dtype not in (torch.float64, torch.float32, torch.float16):
            errmsg = "An invalid `dtype` for the `reward` tensor."
            raise RuntimeError(errmsg)

        _episodes.append(episode.detach().clone().cpu())

    episodes: TensorDict = torch.cat(_episodes).to_tensordict()  # type: ignore

    return episodes
