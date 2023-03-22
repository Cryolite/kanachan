#!/usr/bin/env python3

import math
from collections import OrderedDict
from typing import (Tuple,)
import torch
from torch import nn
from kanachan.training.constants import NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.value_model import ValueDecoder


class QVDecoder(nn.Module):
    def __init__(
            self, *, dimension: int, dim_feedforward: int, activation_function: str, dropout: float,
            num_layers: int, device: torch.device, dtype: torch.dtype) -> None:
        super(QVDecoder, self).__init__()

        self.value_decoder = ValueDecoder(
            dimension=dimension, dim_feedforward=dim_feedforward,
            activation_function=activation_function, dropout=dropout, num_layers=num_layers,
            device=device, dtype=dtype)

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

    def forward(
            self, candidates: torch.Tensor,
            encode: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert candidates.dim() == 2
        assert encode.dim() == 3
        assert candidates.size(0) == encode.size(0)
        
        value: torch.Tensor = self.value_decoder(candidates, encode)
        assert value.dim() == 1
        assert value.size(0) == candidates.size(0)
        value_expanded = torch.unsqueeze(value, dim=1)
        value_expanded = value_expanded.expand(-1, MAX_NUM_ACTION_CANDIDATES)

        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        advantage: torch.Tensor = self.layers(encode)
        advantage = torch.squeeze(advantage, dim=2)
        assert advantage.dim() == 2
        assert advantage.size(0) == candidates.size(0)
        assert advantage.size(1) == MAX_NUM_ACTION_CANDIDATES

        q = value_expanded + advantage
        q = torch.where(candidates < NUM_TYPES_OF_ACTIONS, q, torch.full_like(q, -math.inf))

        return q, value


class QVModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: QVDecoder) -> None:
        super(QVModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self, sparse: torch.Tensor, numeric: torch.Tensor, progression: torch.Tensor,
            candidates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encode = self.encoder(sparse, numeric, progression, candidates)
        return self.decoder(candidates, encode)
