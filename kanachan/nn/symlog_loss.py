import torch
from torch import Tensor, nn


class SymlogLoss(nn.Module):
    def __init__(self, reduction: str='mean') -> None:
        super().__init__()
        self.__reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.sign() * torch.log(target.abs() + 1.0)
        return nn.functional.mse_loss(input, target, reduction=self.__reduction)
