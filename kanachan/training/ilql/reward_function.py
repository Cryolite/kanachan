#!/usr/bin/env python3

from typing import (Optional, Callable,)
import torch


RewardFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
     int, Optional[int], Optional[int]],
    float]
