#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS,)
from kanachan.training.positional_embedding import PositionalEmbedding


class Encoder(nn.Module):
    def __init__(
            self, *, dimension: int, num_heads: int, dim_feedforward: int,
            num_layers:int, dropout: float, activation_function: str,
            checkpointing: bool, **kwargs) -> None:
        super(Encoder, self).__init__()

        self.sparse_embedding = nn.Embedding(
            NUM_TYPES_OF_SPARSE_FEATURES + 1, dimension,
            padding_idx=NUM_TYPES_OF_SPARSE_FEATURES)

        numeric_embedding_initial = torch.normal(
            0.0, 1.0, size=(NUM_NUMERIC_FEATURES, dimension - 1))
        self.numeric_embedding \
            = numeric_embedding_initial.clone().detach().requires_grad_(True)

        self.progression_embedding = PositionalEmbedding(
            NUM_TYPES_OF_PROGRESSION_FEATURES + 1, dimension,
            padding_idx=NUM_TYPES_OF_PROGRESSION_FEATURES,
            max_length=MAX_LENGTH_OF_PROGRESSION_FEATURES)

        self.candidates_embedding = nn.Embedding(
            NUM_TYPES_OF_ACTIONS + 2, dimension,
            padding_idx=NUM_TYPES_OF_ACTIONS + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            dimension, num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation_function,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.checkpointing = checkpointing

    def forward(self, x):
        sparse, numeric, progression, candidates = x
        sparse = self.sparse_embedding(sparse)
        numeric = torch.unsqueeze(numeric, -1)
        numeric_embedding = torch.unsqueeze(self.numeric_embedding, 0)
        numeric_embedding = numeric_embedding.expand(numeric.size(0), -1, -1)
        numeric_embedding = numeric_embedding.to(numeric)
        numeric = torch.cat((numeric, numeric_embedding), -1)
        progression = self.progression_embedding(progression)
        candidates = self.candidates_embedding(candidates)
        embedding = torch.cat(
            (sparse, numeric, progression, candidates), dim=1)
        if self.checkpointing:
            encoder_layers = self.encoder.layers
            encode = checkpoint_sequential(
                encoder_layers, len(encoder_layers), embedding)
        else:
            encode = self.encoder(embedding)
        return encode
