#!/usr/bin/env python3

from collections import OrderedDict
import torch
from torch import nn
from kanachan.training.constants import (
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES,)
from kanachan.training.bert.encoder import Encoder


class ValueDecoder(nn.Module):
    def __init__(
            self, *, dimension: int, dim_feedforward: int, activation_function: str, dropout: float,
            num_layers: int, device: torch.device, dtype: torch.dtype) -> None:
        if num_layers <= 0:
            raise ValueError(num_layers)

        super(ValueDecoder, self).__init__()

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
        layers[f'layer{num_layers - 1}'] = nn.Linear(
            dimension if num_layers == 1 else dim_feedforward, 1, device=device, dtype=dtype)
        self.layers = nn.Sequential(layers)

    def forward(self, candidates: torch.Tensor, encode: torch.Tensor) -> torch.Tensor:
        assert candidates.dim() == 2
        assert encode.dim() == 3
        assert candidates.size(0) == encode.size(0)

        mask = (candidates == NUM_TYPES_OF_ACTIONS)

        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        value: torch.Tensor = self.layers(encode)
        value = torch.squeeze(value, dim=2)
        value = value[mask]
        assert value.dim() == 1
        assert value.size(0) == candidates.size(0)

        return value


class ValueModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: ValueDecoder) -> None:
        super(ValueModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self, sparse: torch.Tensor, numeric: torch.Tensor, progression: torch.Tensor,
            candidates: torch.Tensor) -> torch.Tensor:
        encode = self.encoder(sparse, numeric, progression, candidates)
        prediction = self.decoder(candidates, encode)
        return prediction
