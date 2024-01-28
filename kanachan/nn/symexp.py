from torch import Tensor, nn


class Symexp(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.sign() * (x.abs().exp() - 1.0)
