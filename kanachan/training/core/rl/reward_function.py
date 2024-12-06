from typing import Callable
from tensordict import TensorDict  # type: ignore


RewardFunction = Callable[[TensorDict, bool], None]
