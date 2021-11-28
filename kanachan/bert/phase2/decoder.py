#!/usr/bin/env python3

import torch
from torch import nn
from kanachan.constants import MAX_NUM_ACTION_CANDIDATES


class Decoder(nn.Module):
    def __init__(
            self, *, dimension: int, dim_final_feedforward: int, dropout: float,
            activation_function: str, **kwargs) -> None:
        super(Decoder, self).__init__()

        # The final layer is position-wise feed-forward network.
        self._semifinal_linear = nn.Linear(dimension, dim_final_feedforward)
        self._semifinal_dropout = nn.Dropout(p=dropout)
        if activation_function == 'relu':
            self._semifinal_activation = nn.ReLU()
        elif activation_function == 'gelu':
            self._semifinal_activation = nn.GELU()
        else:
            raise ValueError(
                f'{activation_function}: invalid activation function')
        self._final_linear = nn.Linear(dim_final_feedforward, 1)

        self.__mode = 'training'

    def mode(self, mode: str) -> None:
        if mode not in ('training', 'validation', 'prediction'):
            raise ValueError(mode)
        self.__mode = mode

    def forward(self, encode):
        if self.__mode == 'prediction':
            encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        else:
            encode, index = encode
            assert(torch.all(0 <= index))
            assert(torch.all(index < MAX_NUM_ACTION_CANDIDATES))
            rows = torch.arange(encode.size()[0])
            encode = encode[rows, -MAX_NUM_ACTION_CANDIDATES + index]

        decode = self._semifinal_linear(encode)
        decode = self._semifinal_dropout(decode)
        decode = self._semifinal_activation(decode)

        prediction = self._final_linear(decode)
        prediction = torch.squeeze(prediction, dim=-1)
        return prediction
