from typing import Callable, Tuple, Union, List
import torch


RewardFunction = Callable[
    [
        Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            int, int, bool]],
    Union[torch.Tensor, List[float]]]
