#!/usr/bin/env python3

import torch
from torch import nn
from kanachan.training.constants import (
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES,)
from kanachan.training.bert.encoder import Encoder


class ValueDecoder(nn.Module):
    def __init__(
            self, *, dimension: int, dim_final_feedforward: int,
            activation_function: str, dropout: float, **kwargs) -> None:
        super(ValueDecoder, self).__init__()

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

    def forward(self, x) -> torch.Tensor:
        candidates, encode = x

        new_candidates = candidates.clone().detach()
        for i in range(new_candidates.size(0)):
            for j in range(new_candidates.size(1)):
                if new_candidates[i, j] == NUM_TYPES_OF_ACTIONS:
                    break
                if new_candidates[i, j] == NUM_TYPES_OF_ACTIONS + 1:
                    new_candidates[i, j] = NUM_TYPES_OF_ACTIONS
                    break

        mask = (new_candidates == NUM_TYPES_OF_ACTIONS)

        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        decode = self._semifinal_linear(encode)
        decode = self._semifinal_activation(decode)
        decode = self._semifinal_dropout(decode)
        prediction = self._final_linear(decode)
        prediction = torch.squeeze(prediction, dim=2)
        prediction = prediction[mask]
        assert(prediction.dim() == 1)
        assert(prediction.size(0) == new_candidates.size(0))

        return prediction


class ValueModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: ValueDecoder) -> None:
        super(ValueModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x) -> torch.Tensor:
        encode = self._encoder(x)
        prediction = self._decoder((x[3], encode))
        return prediction
