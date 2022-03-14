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
        self.__semifinal_linear = nn.Linear(dimension, dim_final_feedforward)
        if activation_function == 'relu':
            self.__semifinal_activation = nn.ReLU()
        elif activation_function == 'gelu':
            self.__semifinal_activation = nn.GELU()
        else:
            raise ValueError(
                f'{activation_function}: An invalid activation function.')
        self.__semifinal_dropout = nn.Dropout(p=dropout)
        self.__final_linear = nn.Linear(dim_final_feedforward, 1)

    def forward(self, x) -> torch.Tensor:
        candidates, encode = x

        indices = torch.nonzero(candidates == NUM_TYPES_OF_ACTIONS)

        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        encode = encode[indices[:, 0], indices[:, 1]]
        decode = self.__semifinal_linear(encode)
        decode = self.__semifinal_activation(decode)
        decode = self.__semifinal_dropout(decode)

        prediction = self.__final_linear(decode)
        prediction = torch.squeeze(prediction, dim=1)
        return prediction


class ValueModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: ValueDecoder) -> None:
        super(ValueModel, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder

    def forward(self, x) -> torch.Tensor:
        encode = self.__encoder(x)
        prediction = self.__decoder((x[3], encode))
        return prediction
