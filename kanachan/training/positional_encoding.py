#!/usr/bin/env python3

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_length: int, dimension: int, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        pe = torch.zeros(max_length, dimension)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

    def forward(self, x):
        x = x + self._pe[:x.size(1), :]
        return self.dropout(x)
