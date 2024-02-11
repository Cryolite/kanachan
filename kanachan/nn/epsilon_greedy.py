import torch
from torch import Tensor
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from kanachan.constants import NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES


class EpsilonGreedy(TensorDictModuleBase):
    def __init__(
        self,
        *,
        epsilon_first: float,
        epsilon_last: float,
        annealing_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if epsilon_first < 0.0 or epsilon_first > 1.0:
            msg = (
                "`epsilon_first` must be a real number within the range"
                " [0.0, 1.0]."
            )
            raise ValueError(msg)
        if epsilon_last > epsilon_first:
            msg = (
                "`epsilon_last` must be less than or equal to"
                " `epsilon_last`."
            )
            raise ValueError(msg)
        if epsilon_last < 0.0:
            msg = "`epsilon_last` must be a non-negative real number."
            raise ValueError(msg)
        if annealing_steps < 0:
            msg = "`annealing_steps` must be a non-negative integer."
            raise ValueError(msg)
        if annealing_steps == 0 and epsilon_first != epsilon_last:
            msg = (
                "`annealing_steps == 0` is possible only if"
                " `epsilon_first == epsilon_last`."
            )
            raise ValueError(msg)

        super().__init__()

        self.register_buffer(
            "epsilon_first",
            torch.tensor(epsilon_first, device=device, dtype=dtype),
        )
        self.register_buffer(
            "epsilon_last",
            torch.tensor(epsilon_last, device=device, dtype=dtype),
        )
        self.register_buffer(
            "epsilon", torch.tensor(epsilon_first, device=device, dtype=dtype)
        )
        self.annealing_steps = annealing_steps
        self._step = 0

        self.in_keys = ["candidates", "action"]
        self.out_keys = ["action"]

    def forward(self, td: TensorDict) -> TensorDict:
        batch_size = td.size(0)
        assert isinstance(batch_size, int)
        candidates: Tensor = td["candidates"]
        assert candidates.dim() == 2
        assert candidates.size(0) == batch_size
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
        action: Tensor = td["action"]
        assert action.dim() == 1
        assert action.size(0) == batch_size

        if not self.training:
            return td

        action_mask = candidates < NUM_TYPES_OF_ACTIONS
        random_action = (
            action_mask.float().multinomial(1).squeeze(1).to(dtype=torch.int32)
        )
        assert random_action.dim() == 1
        assert random_action.size(0) == batch_size
        assert (
            (
                candidates[torch.arange(batch_size), random_action]
                < NUM_TYPES_OF_ACTIONS
            )
            .all()
            .item()
        )

        random = torch.rand_like(action, dtype=torch.float32)
        epsilon_mask = random < self.epsilon

        td["action"] = torch.where(epsilon_mask, random_action, action)

        return td

    def step(self, steps: int = 1) -> None:
        if steps < 0:
            msg = "`steps` must be a non-negative integer."
            raise ValueError(msg)

        if self._step + steps >= self.annealing_steps:
            self._step = self.annealing_steps

        epsilon_first: Tensor = self.epsilon_first
        epsilon_last: Tensor = self.epsilon_last
        if self.annealing_steps == 0:
            assert epsilon_first.item() == epsilon_last.item()
            return

        # pylint: disable=attribute-defined-outside-init
        self.epsilon = epsilon_first * (
            (self.annealing_steps - self._step) / self.annealing_steps
        )
        self.epsilon += epsilon_last * (self._step / self.annealing_steps)
