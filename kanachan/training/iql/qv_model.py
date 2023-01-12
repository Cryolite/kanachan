#!/usr/bin/env python3

import math
from typing import (Tuple,)
import torch
from torch import nn
from kanachan.training.constants import (
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES,)
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.value_model import ValueDecoder


class QVDecoder(nn.Module):
    def __init__(
            self, *, dimension: int, dim_final_feedforward: int,
            activation_function: str, dropout: float, **kwargs) -> None:
        super(QVDecoder, self).__init__()

        self._value_decoder = ValueDecoder(
            dimension=dimension, dim_final_feedforward=dim_final_feedforward,
            activation_function=activation_function, dropout=dropout, **kwargs)

        # The final layer is position-wise feed-forward network.
        self._semifinal_linear = nn.Linear(dimension, dim_final_feedforward)
        if activation_function == 'relu':
            self._semifinal_activation = nn.ReLU()
        elif activation_function == 'gelu':
            self._semifinal_activation = nn.GELU()
        else:
            raise ValueError(
                f'{activation_function}: An invalid activation function.')
        self._semifinal_dropout = nn.Dropout(p=dropout)
        self._final_linear = nn.Linear(dim_final_feedforward, 1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        candidates, encode = x

        value_decode = self._value_decoder(x)
        value_expanded = torch.unsqueeze(value_decode, dim=1)
        value_expanded = value_expanded.expand(-1, MAX_NUM_ACTION_CANDIDATES)

        advantage_encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        advantage_decode = self._semifinal_linear(advantage_encode)
        advantage_decode = self._semifinal_activation(advantage_decode)
        advantage_decode = self._semifinal_dropout(advantage_decode)
        advantage_decode = self._final_linear(advantage_decode)
        advantage_decode = torch.squeeze(advantage_decode, dim=2)
        assert(advantage_decode.dim() == 2)
        assert(advantage_decode.size(0) == candidates.size(0))
        assert(advantage_decode.size(1) == MAX_NUM_ACTION_CANDIDATES)

        q_decode = value_expanded + advantage_decode
        q_decode = torch.where(
            candidates < NUM_TYPES_OF_ACTIONS, q_decode, -math.inf)

        return (q_decode, value_decode)


class QVModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: QVDecoder) -> None:
        super(QVModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        encode = self._encoder(x)
        return self._decoder((x[3], encode))
