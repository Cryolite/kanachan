#!/usr/bin/env python3

import torch
from torch import nn
from kanachan.constants import (
    NUM_SPARSE_FEATURES, NUM_TYPES_OF_POSITIONAL_FEATURES,
    MAX_LENGTH_OF_POSITIONAL_FEATURES, NUM_ACTIONS,)
from kanachan.positional_embedding import PositionalEmbedding


class Encoder(nn.Module):
    def __init__(
            self, num_dimensions: int, num_heads: int, dim_feedforward: int,
            num_layers:int, dropout: float=0.1, activation_function='gelu',
            sparse: bool=False) -> None:
        super(Encoder, self).__init__()

        self.__sparse_embedding = nn.Embedding(
            NUM_SPARSE_FEATURES + 1, num_dimensions,
            padding_idx=NUM_SPARSE_FEATURES, sparse=sparse)
        self.__sparse_dropout = nn.Dropout(p=dropout)

        self.__positional_embedding = PositionalEmbedding(
            NUM_TYPES_OF_POSITIONAL_FEATURES + 1, num_dimensions,
            padding_idx=NUM_TYPES_OF_POSITIONAL_FEATURES,
            max_length=MAX_LENGTH_OF_POSITIONAL_FEATURES, dropout=dropout,
            sparse=sparse)

        self.__candidates_embedding = nn.Embedding(
            NUM_ACTIONS + 1, num_dimensions, padding_idx=NUM_ACTIONS,
            sparse=sparse)
        self.__candidates_dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            num_dimensions, num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation_function, batch_first=True)
        self.__encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        sparse, numeric, positional, candidates = x
        sparse = self.__sparse_embedding(sparse)
        sparse = self.__sparse_dropout(sparse)
        positional = self.__positional_embedding(positional)
        candidates = self.__candidates_embedding(candidates)
        candidates = self.__candidates_dropout(candidates)
        embedding = torch.cat(
            (sparse, numeric, positional, candidates), dim=1)
        encode = self.__encoder(embedding)
        return encode
