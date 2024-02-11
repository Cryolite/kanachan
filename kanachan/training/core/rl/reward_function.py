from typing import Callable
from tensordict import TensorDict


RewardFunction = Callable[[TensorDict, bool], None]
