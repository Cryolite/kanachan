#!/usr/bin/env python3

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(
            self, *, max_length: int, dimension: int, dropout: float,
            device: torch.device, dtype: torch.dtype):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dimension, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dimension))
        pe = torch.zeros(max_length, dimension, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        x += self._pe
        return self.dropout(x)
