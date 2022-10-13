#!/usr/bin/env python3

import math
import torch
from torch import nn
from kanachan.training.constants import (
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES,)
from kanachan.training.bert.encoder import Encoder


class PolicyDecoder(nn.Module):
    def __init__(
            self, *, dimension: int, dim_final_feedforward: int,
            activation_function: str, dropout: float, **kwargs) -> None:
        super(PolicyDecoder, self).__init__()

        self.dimension = dimension

        # The final layer is position-wise feed-forward network.
        self.semifinal_linear = nn.Linear(dimension, dim_final_feedforward)
        if activation_function == 'relu':
            self.semifinal_activation = nn.ReLU()
        elif activation_function == 'gelu':
            self.semifinal_activation = nn.GELU()
        else:
            raise ValueError(
                f'{activation_function}: invalid activation function')
        self.semifinal_dropout = nn.Dropout(p=dropout)
        self.final_linear = nn.Linear(dim_final_feedforward, 1)

    def forward(self, x):
        candidates, encode = x

        mask = (candidates < NUM_TYPES_OF_ACTIONS)
        mask = torch.unsqueeze(mask, dim=2)
        mask = mask.expand(-1, -1, self.dimension)
        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:] * mask
        decode = self.semifinal_linear(encode)
        decode = self.semifinal_activation(decode)
        decode = self.semifinal_dropout(decode)

        prediction = self.final_linear(decode)
        prediction = torch.squeeze(prediction, dim=2)
        mask = torch.tensor(
            -math.inf, device=prediction.device, dtype=prediction.dtype)
        prediction = torch.where(
            candidates < NUM_TYPES_OF_ACTIONS, prediction, mask)
        return prediction


class PolicyModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: PolicyDecoder) -> None:
        super(PolicyModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def mode(self, mode: str) -> None:
        if mode not in ('training', 'validation', 'prediction'):
            raise ValueError(mode)

    def forward(self, x) -> torch.Tensor:
        encode = self.encoder(x)
        decode = self.decoder((x[3], encode))
        return decode
