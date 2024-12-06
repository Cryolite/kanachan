import math
from typing import Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchrl.data.utils import DEVICE_TYPING  # type: ignore


class NoisyLinear(nn.Module):
    """Noisy Linear Layer.

    Presented in "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

    A Noisy Linear Layer is a linear layer with parametric noise added
    to the weights. This induced stochasticity can be used in RL
    networks for the agent's policy to aid efficient exploration. The
    parameters of the noise are learned with gradient descent along with
    any other remaining network weights. Factorized Gaussian noise is
    the type of noise usually employed.

    Args:
        in_features: input features dimension
        out_features: out features dimension
        bias: if ``True``, a bias term will be added to the matrix
            multiplication: Ax + b. Defaults to ``True``
        device: device of the layer. Defaults to ``"cpu"``
        dtype: dtype of the parameters. Defaults to ``None`` (default
            pytorch dtype)
        std_init: initial value of the Gaussian standard deviation
            before optimization. Defaults to ``0.1``

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: DEVICE_TYPING | None = None,
        dtype: torch.dtype | None = None,
        std_init: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        self.bias_mu: nn.Parameter | None
        if bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(
                self.std_init / math.sqrt(self.out_features)
            )

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(
        self, size: int | torch.Size | Sequence[int]
    ) -> torch.Tensor:
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(*size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: Tensor) -> Tensor:
        weight: Tensor
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        else:
            weight = self.weight_mu

        bias: Tensor
        if self.bias_mu is not None:
            if self.training:
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                bias = self.bias_mu
        else:
            bias = torch.zeros_like(x)

        return F.linear(x, weight, bias)  # pylint: disable=not-callable
