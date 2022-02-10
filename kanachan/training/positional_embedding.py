#!/usr/bin/env python3

from typing import Optional
from torch import nn
from kanachan.training.positional_encoding import PositionalEncoding


class PositionalEmbedding(nn.Module):
    def __init__(
            self, num_embeddings: int, embedding_dim:int,
            padding_idx: Optional[int]=None, max_length: int=10000,
            sparse: bool=False):
        super(PositionalEmbedding, self).__init__()
        self.__embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx,
            sparse=sparse)
        self.__positional_encoding = PositionalEncoding(
            max_length, embedding_dim)

    def forward(self, x):
        x = self.__embedding(x)
        x = self.__positional_encoding(x)
        return x
