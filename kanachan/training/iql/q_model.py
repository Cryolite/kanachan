#!/usr/bin/env python3

import numpy
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

        self.__dimension = dimension

        self.__value_decoder = ValueDecoder(
            dimension=dimension, dim_final_feedforward=dim_final_feedforward,
            activation_function=activation_function, dropout=dropout, **kwargs)

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

        mask = (candidates < NUM_TYPES_OF_ACTIONS)
        value_decode = self.__value_decoder(x)
        value_decode = torch.unsqueeze(value_decode, dim=1)
        value_decode = value_decode.expand(-1, MAX_NUM_ACTION_CANDIDATES)
        value_decode = value_decode * mask

        mask = torch.unsqueeze(mask, dim=2)
        mask = mask.expand(-1, -1, self.__dimension)
        advantage_encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:] * mask
        advantage_decode = self.__semifinal_linear(advantage_encode)
        advantage_decode = self.__semifinal_activation(advantage_decode)
        advantage_decode = self.__semifinal_dropout(advantage_decode)
        advantage_decode = self.__final_linear(advantage_decode)
        advantage_decode = torch.squeeze(advantage_decode, dim=2)

        return value_decode + advantage_decode


class QModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: QDecoder) -> None:
        super(QModel, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder

    def forward(self, x) -> torch.Tensor:
        encode = self.__encoder(x)
        decode = self.__decoder((x[3], encode))
        return decode
