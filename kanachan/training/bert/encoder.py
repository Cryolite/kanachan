import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES, NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES
)
from kanachan.training.positional_encoding import PositionalEncoding
from kanachan.training.position_embedding import PositionEmbedding


class Encoder(nn.Module):
    def __init__(
            self, *, position_encoder: str, dimension: int, num_heads: int, dim_feedforward: int,
            num_layers: int, activation_function: str, dropout: float, checkpointing: bool,
            device: torch.device, dtype: torch.dtype) -> None:
        if position_encoder not in ('positional_encoding', 'position_embedding'):
            raise ValueError(position_encoder)
        if dimension < 1:
            raise ValueError(dimension)
        if num_heads < 1:
            raise ValueError(num_heads)
        if dim_feedforward < 1:
            raise ValueError(dim_feedforward)
        if num_layers < 1:
            raise ValueError(num_layers)
        if activation_function not in ('relu', 'gelu'):
            raise ValueError(activation_function)
        if dropout < 0.0 or 1.0 <= dropout:
            raise ValueError(dropout)

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
        else:
            raise NotImplementedError(position_encoder)

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
            candidates: torch.Tensor) -> torch.Tensor:
        assert sparse.dim() == 2
        assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
        assert numeric.dim() == 2
        assert numeric.size(1) == NUM_NUMERIC_FEATURES
        assert progression.dim() == 2
        assert progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
        assert candidates.dim() == 2
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
        assert sparse.size(0) == numeric.size(0)
        assert sparse.size(0) == progression.size(0)
        assert sparse.size(0) == candidates.size(0)

        sparse = self.sparse_embedding(sparse)

        numeric = torch.unsqueeze(numeric, 2)
        numeric_embedding = torch.unsqueeze(self.numeric_embedding, 0)
        numeric_embedding = numeric_embedding.expand(numeric.size(0), -1, -1)
        numeric = torch.cat((numeric, numeric_embedding), 2)

        progression = self.progression_embedding(progression)
        progression = self.position_encoder(progression)

        candidates = self.candidates_embedding(candidates)

        embedding = torch.cat((sparse, numeric, progression, candidates), 1)

        if self.checkpointing:
            encoder_layers = self.encoder.layers
            encode = checkpoint_sequential(encoder_layers, len(encoder_layers), embedding)
        else:
            encode = self.encoder(embedding)

        return encode
