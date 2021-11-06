#!/usr/bin/env python3

from torch import nn


class DecoderBase(nn.Module):
    def __init__(
            self, *, dimension: int, dim_final_feedforward: int, dropout: float,
            activation_function, **kwargs) -> None:
        super(TrainingDecoder, self).__init__()

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

    def forward(self, encode):
        raise NotImplemented
