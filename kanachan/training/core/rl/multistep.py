import torch
from torch import Tensor
from tensordict import TensorDict
from kanachan.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES, NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES, NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES, MAX_NUM_ROUND_SUMMARY, RL_NUM_RESULTS)


def make_multistep(episodes: TensorDict, discount_factor: float, td_steps: int) -> None:
    length: int = episodes.size(0)
    sparse: Tensor = episodes['sparse']
    assert sparse.dim() == 2
    assert sparse.size(0) == length
    assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
    assert sparse.dtype == torch.int32
    numeric: Tensor = episodes['numeric']
    assert numeric.dim() == 2
    assert numeric.size(0) == length
    assert numeric.size(1) == NUM_NUMERIC_FEATURES
    assert numeric.dtype == torch.int32
    progression: Tensor = episodes['progression']
    assert progression.dim() == 2
    assert progression.size(0) == length
    assert progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
    assert progression.dtype == torch.int32
    candidates: Tensor = episodes['candidates']
    assert candidates.dim() == 2
    assert candidates.size(0) == length
    assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
    assert candidates.dtype == torch.int32
    round_summary: Tensor = episodes['next', 'round_summary']
    assert round_summary.dim() == 2
    assert round_summary.size(0) == length
    assert round_summary.size(1) == MAX_NUM_ROUND_SUMMARY
    assert round_summary.dtype == torch.int32
    results: Tensor = episodes['next', 'results']
    assert results.dim() == 2
    assert results.size(0) == length
    assert results.size(1) == RL_NUM_RESULTS
    assert results.dtype == torch.int32
    end_of_round: Tensor = episodes['next', 'end_of_round']
    assert end_of_round.dim() == 1
    assert end_of_round.size(0) == length
    assert end_of_round.dtype == torch.bool
    end_of_game: Tensor = episodes['next', 'end_of_game']
    assert end_of_game.dim() == 1
    assert end_of_game.size(0) == length
    assert end_of_game.dtype == torch.bool
    reward: Tensor = episodes['next', 'reward']
    assert reward.dim() == 1
    assert reward.size(0) == length
    assert reward.dtype in (torch.float64, torch.float32, torch.float16)
    done: Tensor = episodes['next', 'done']
    assert done.dim() == 1
    assert done.size(0) == length
    assert done.dtype == torch.bool

    if discount_factor < 0.0 or 1.0 < discount_factor:
        raise ValueError(
            f'{discount_factor}: '
            '`discount_factor` must be a real number within the range [0.0, 1.0].')
    if td_steps <= 0:
        raise ValueError(f'{td_steps}: `td_steps` must be a positive integer.')

    for t in range(length):
        modified_reward: float = reward[t].item()
        _done = False
        if done[t].item():
            _done = True
        else:
            _discount_factor = discount_factor
            for tt in range(1, td_steps):
                modified_reward += _discount_factor * reward[t + tt].item()
                if done[t + tt].item():
                    _done = True
                    break
                _discount_factor *= discount_factor

        _round_summary: Tensor = round_summary[t].detach().clone()
        _results: Tensor = results[t].detach().clone()
        _end_of_round: bool = end_of_round[t].item()
        _end_of_game: bool = end_of_game[t].item()
        if t + td_steps - 1 < length:
            episodes['next'][t] = episodes['next'][t + td_steps - 1]
        episodes['next', 'round_summary'][t] = _round_summary
        episodes['next', 'results'][t] = _results
        episodes['next', 'end_of_round'][t] = _end_of_round
        episodes['next', 'end_of_game'][t] = _end_of_game
        if _done:
            episodes['next', 'sparse'][t] \
                = torch.full_like(sparse[t], NUM_TYPES_OF_SPARSE_FEATURES)
            episodes['next', 'numeric'][t] = torch.zeros_like(numeric[t])
            episodes['next', 'progression'][t] \
                = torch.full_like(progression[t], NUM_TYPES_OF_PROGRESSION_FEATURES)
            episodes['next', 'candidates'][t] \
                = torch.full_like(candidates[t], NUM_TYPES_OF_ACTIONS)
            episodes['next', 'reward'][t] = modified_reward
            episodes['next', 'done'][t] = _done
