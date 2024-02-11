import torch
from torch import nn
from kanachan.constants import (
    NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES,
    ENCODER_WIDTH,
)
from kanachan.nn import Encoder, MlpDecoder


class PolicyDecoder(nn.Module):
    def __init__(
        self,
        *,
        dimension: int,
        dim_feedforward: int,
        activation_function: str,
        dropout: float,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if dimension < 1:
            raise ValueError(dimension)
        if dim_feedforward < 1:
            raise ValueError(dim_feedforward)
        if activation_function not in ("relu", "gelu"):
            raise ValueError(activation_function)
        if dropout < 0.0 or 1.0 <= dropout:
            raise ValueError(dropout)
        if num_layers < 1:
            raise ValueError(num_layers)

        super().__init__()

        self.linear_decoder = MlpDecoder(
            dimension=dimension,
            dim_feedforward=dim_feedforward,
            activation_function=activation_function,
            dropout=dropout,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )

    def forward(
        self, candidates: torch.Tensor, encode: torch.Tensor
    ) -> torch.Tensor:
        assert candidates.dim() == 2
        batch_size = candidates.size(0)
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
        assert encode.dim() == 3
        assert encode.size(0) == batch_size
        assert encode.size(1) == ENCODER_WIDTH

        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        decode = self.linear_decoder(encode)
        prediction = nn.Softmax(dim=1)(decode)
        prediction = torch.where(
            candidates < NUM_TYPES_OF_ACTIONS, prediction, 0.0
        )

        return prediction


class PolicyModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: PolicyDecoder) -> None:
        super(PolicyModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        sparse: torch.Tensor,
        numeric: torch.Tensor,
        progression: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        encode = self.encoder(sparse, numeric, progression, candidates)
        prediction = self.decoder(candidates, encode)

        return prediction
