#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from kanachan.constants import (
    NUM_SPARSE_FEATURES, NUM_TYPES_OF_POSITIONAL_FEATURES,
    MAX_LENGTH_OF_POSITIONAL_FEATURES, NUM_ACTIONS,)
from kanachan.positional_embedding import PositionalEmbedding


class Encoder(nn.Module):
    def __init__(
            self, *, dimension: int, num_heads: int, dim_feedforward: int,
            num_layers:int, dropout: float, activation_function,
            checkpointing: bool, **kwargs) -> None:
        super(Encoder, self).__init__()

        self.__sparse_embedding = nn.Embedding(
            NUM_SPARSE_FEATURES + 1, dimension,
            padding_idx=NUM_SPARSE_FEATURES)
        self.__sparse_dropout = nn.Dropout(p=dropout)

        self.__positional_embedding = PositionalEmbedding(
            NUM_TYPES_OF_POSITIONAL_FEATURES + 1, dimension,
            padding_idx=NUM_TYPES_OF_POSITIONAL_FEATURES,
            max_length=MAX_LENGTH_OF_POSITIONAL_FEATURES, dropout=dropout)

        self.__candidates_embedding = nn.Embedding(
            NUM_ACTIONS + 1, dimension, padding_idx=NUM_ACTIONS)
        self.__candidates_dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            dimension, num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation_function,
            batch_first=True)
        self.__encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.__checkpointing = checkpointing

    def forward(self, x):
        sparse, numeric, positional, candidates = x
        sparse = self.__sparse_embedding(sparse)
        sparse = self.__sparse_dropout(sparse)
        positional = self.__positional_embedding(positional)
        candidates = self.__candidates_embedding(candidates)
        candidates = self.__candidates_dropout(candidates)
        embedding = torch.cat(
            (sparse, numeric, positional, candidates), dim=1)
        if self.__checkpointing:
            encoder_layers = self.__encoder.layers
            encode = checkpoint_sequential(
                encoder_layers, len(encoder_layers), embedding)
        else:
            encode = self.__encoder(embedding)
        return encode
