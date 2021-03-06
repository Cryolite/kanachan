#!/usr/bin/env python3

import torch
from torch import nn
from kanachan.training.constants import MAX_NUM_ACTION_CANDIDATES


class Decoder(nn.Module):
    def __init__(
            self, *, dimension: int, dim_final_feedforward: int,
            dropout: float, activation_function: str, **kwargs) -> None:
        super(Decoder, self).__init__()

        # The final layer is position-wise feed-forward network.
        self.__semifinal_linear = nn.Linear(dimension, dim_final_feedforward)
        self.__semifinal_dropout = nn.Dropout(p=dropout)
        if activation_function == 'relu':
            self.__semifinal_activation = nn.ReLU()
        elif activation_function == 'gelu':
            self.__semifinal_activation = nn.GELU()
        else:
            raise ValueError(
                f'{activation_function}: invalid activation function')
        self.__final_linear = nn.Linear(dim_final_feedforward, 1)

    def forward(self, encode):
        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        decode = self.__semifinal_linear(encode)
        decode = self.__semifinal_dropout(decode)
        decode = self.__semifinal_activation(decode)

        prediction = self.__final_linear(decode)
        prediction = torch.squeeze(prediction, dim=2)
        return prediction
