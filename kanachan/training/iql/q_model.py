#!/usr/bin/env python3

import math
import torch
from torch import nn
from kanachan.training.constants import (
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES,)
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.value_model import ValueDecoder


class QDecoder(nn.Module):
    def __init__(
            self, *, dimension: int, dim_final_feedforward: int,
            activation_function: str, dropout: float, **kwargs) -> None:
        super(QDecoder, self).__init__()

        self.value_decoder = ValueDecoder(
            dimension=dimension, dim_final_feedforward=dim_final_feedforward,
            activation_function=activation_function, dropout=dropout, **kwargs)

        # The final layer is position-wise feed-forward network.
        self.semifinal_linear = nn.Linear(dimension, dim_final_feedforward)
        if activation_function == 'relu':
            self.semifinal_activation = nn.ReLU()
        elif activation_function == 'gelu':
            self.semifinal_activation = nn.GELU()
        else:
            raise ValueError(
                f'{activation_function}: An invalid activation function.')
        self.semifinal_dropout = nn.Dropout(p=dropout)
        self.final_linear = nn.Linear(dim_final_feedforward, 1)

    def forward(self, x) -> torch.Tensor:
        candidates, encode = x

        value_decode = self.value_decoder(x)
        value_decode = torch.unsqueeze(value_decode, dim=1)
        value_decode = value_decode.expand(-1, MAX_NUM_ACTION_CANDIDATES)

        advantage_encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        advantage_decode = self.semifinal_linear(advantage_encode)
        advantage_decode = self.semifinal_activation(advantage_decode)
        advantage_decode = self.semifinal_dropout(advantage_decode)
        advantage_decode = self.final_linear(advantage_decode)
        advantage_decode = torch.squeeze(advantage_decode, dim=2)
        assert(advantage_decode.dim() == 2)
        assert(advantage_decode.size(0) == candidates.size(0))
        assert(advantage_decode.size(1) == MAX_NUM_ACTION_CANDIDATES)
        prediction = value_decode + advantage_decode
        prediction = torch.where(
            candidates < NUM_TYPES_OF_ACTIONS, prediction, -math.inf)

        return prediction


class QModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: QDecoder) -> None:
        super(QModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x) -> torch.Tensor:
        encode = self.encoder(x)
        decode = self.decoder((x[3], encode))
        return decode
