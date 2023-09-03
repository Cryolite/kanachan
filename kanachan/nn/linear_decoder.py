import math
from collections import OrderedDict
import torch
import torch.nn as nn
from kanachan.training.constants import NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES


class LinearDecoder(nn.Module):
    def __init__(
            self, *, dimension: int, dim_feedforward: int, activation_function: str, dropout: float,
            num_layers: int, device: torch.device, dtype: torch.dtype) -> None:
        if dimension < 1:
            raise ValueError(dimension)
        if dim_feedforward < 1:
            raise ValueError(dim_feedforward)
        if activation_function not in ('relu', 'gelu',):
            raise ValueError(activation_function)
        if dropout < 0.0 or 1.0 <= dropout:
            raise ValueError(dropout)
        if num_layers < 1:
            raise ValueError(num_layers)

        super().__init__()

        layers = OrderedDict()
        for i in range(num_layers - 1):
            layers[f'layer{i}'] = nn.Linear(
                dimension if i == 0 else dim_feedforward, dim_feedforward,
                device=device, dtype=dtype)
            if activation_function == 'relu':
                layers[f'activation{i}'] = nn.ReLU()
            elif activation_function == 'gelu':
                layers[f'activation{i}'] = nn.GELU()
            else:
                raise ValueError(activation_function)
            layers[f'dropout{i}'] = nn.Dropout(p=dropout)
        suffix = '' if num_layers == 1 else str(num_layers - 1)
        layers['layer' + suffix] = nn.Linear(
            dimension if num_layers == 1 else dim_feedforward, 1, device=device, dtype=dtype)
        self.layers = nn.Sequential(layers)

    def forward(self, candidates: torch.Tensor, encode: torch.Tensor) -> torch.Tensor:
        assert candidates.dim() == 2
        assert encode.dim() == 3
        assert encode.size(1) == MAX_NUM_ACTION_CANDIDATES
        assert candidates.size(0) == encode.size(0)

        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        output: torch.Tensor = self.layers(encode)
        output = torch.squeeze(output, dim=2)
        assert output.dim() == 2
        assert output.size(0) == candidates.size(0)
        assert output.size(1) == MAX_NUM_ACTION_CANDIDATES
        output = torch.where(candidates < NUM_TYPES_OF_ACTIONS, output, -math.inf)

        return output
