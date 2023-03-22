#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS
)
from kanachan.training.positional_encoding import PositionalEncoding
from kanachan.training.position_embedding import PositionEmbedding


class Encoder(nn.Module):
    def __init__(
            self, *, position_encoder: str, dimension: int, num_heads: int, dim_feedforward: int,
            num_layers: int, activation_function: str, dropout: float, checkpointing: bool,
            device: torch.device, dtype: torch.dtype) -> None:
        super(Encoder, self).__init__()

        self.sparse_embedding = nn.Embedding(
            NUM_TYPES_OF_SPARSE_FEATURES + 1, dimension, padding_idx=NUM_TYPES_OF_SPARSE_FEATURES,
            device=device, dtype=dtype)

        self.numeric_embedding = nn.Parameter(
            torch.randn(NUM_NUMERIC_FEATURES, dimension - 1, device=device, dtype=dtype))

        self.progression_embedding = nn.Embedding(
            NUM_TYPES_OF_PROGRESSION_FEATURES + 1, dimension,
            padding_idx=NUM_TYPES_OF_PROGRESSION_FEATURES, device=device, dtype=dtype)
        if position_encoder == 'positional_encoding':
            self.position_encoder = PositionalEncoding(
                max_length=MAX_LENGTH_OF_PROGRESSION_FEATURES, dimension=dimension, dropout=dropout,
                device=device, dtype=dtype)
        elif position_encoder == 'position_embedding':
            self.position_encoder = PositionEmbedding(
                max_length=MAX_LENGTH_OF_PROGRESSION_FEATURES, dimension=dimension, dropout=dropout,
                device=device, dtype=dtype)

        self.candidates_embedding = nn.Embedding(
            NUM_TYPES_OF_ACTIONS + 2, dimension, padding_idx=NUM_TYPES_OF_ACTIONS + 1,
            device=device, dtype=dtype)

        encoder_layer = nn.TransformerEncoderLayer(
            dimension, num_heads, dim_feedforward=dim_feedforward, activation=activation_function,
            dropout=dropout, batch_first=True, device=device, dtype=dtype)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.checkpointing = checkpointing

    def forward(
            self, sparse: torch.Tensor, numeric: torch.Tensor, progression: torch.Tensor,
            candidates: torch.Tensor):
        sparse = self.sparse_embedding(sparse)

        numeric = torch.unsqueeze(numeric, -1)
        numeric_embedding = torch.unsqueeze(self.numeric_embedding, 0)
        numeric_embedding = numeric_embedding.expand(numeric.size(0), -1, -1)
        numeric = torch.cat((numeric, numeric_embedding), -1)

        progression = self.progression_embedding(progression)
        progression = self.position_encoder(progression)

        candidates = self.candidates_embedding(candidates)

        embedding = torch.cat((sparse, numeric, progression, candidates), dim=1)

        if self.checkpointing:
            encoder_layers = self.encoder.layers
            encode = checkpoint_sequential(encoder_layers, len(encoder_layers), embedding)
        else:
            encode = self.encoder(embedding)

        return encode
