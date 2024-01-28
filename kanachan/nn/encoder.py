import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint_sequential
from kanachan.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTIVE_SPARSE_FEATURES, NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES, MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES
)
from kanachan import piecewise_linear_encoding
from kanachan.nn import PositionalEncoding, PositionEmbedding


class Encoder(nn.Module):
    def __init__(
            self, *, position_encoder: str, dimension: int, num_heads: int, dim_feedforward: int,
            activation_function: str, dropout: float, num_layers: int, checkpointing: bool,
            device: torch.device, dtype: torch.dtype) -> None:
        if position_encoder not in ('positional_encoding', 'position_embedding'):
            raise ValueError(position_encoder)
        if dimension <= 0:
            raise ValueError(dimension)
        if num_heads <= 0:
            raise ValueError(num_heads)
        if dim_feedforward <= 0:
            raise ValueError(dim_feedforward)
        if num_layers <= 0:
            raise ValueError(num_layers)
        if activation_function not in ('relu', 'gelu'):
            raise ValueError(activation_function)
        if dropout < 0.0 or 1.0 <= dropout:
            raise ValueError(dropout)

        super().__init__()

        self.__dimension = dimension

        self.sparse_embedding = nn.Embedding(
            NUM_TYPES_OF_SPARSE_FEATURES + 1, self.__dimension,
            padding_idx=NUM_TYPES_OF_SPARSE_FEATURES, device=device, dtype=dtype)

        self.numeric_embedding = nn.Embedding(
            NUM_NUMERIC_FEATURES, self.__dimension // 2, device=device, dtype=dtype)

        self.progression_embedding = nn.Embedding(
            NUM_TYPES_OF_PROGRESSION_FEATURES + 1, self.__dimension,
            padding_idx=NUM_TYPES_OF_PROGRESSION_FEATURES, device=device, dtype=dtype)
        if position_encoder == 'positional_encoding':
            self.position_encoder = PositionalEncoding(
                max_length=MAX_LENGTH_OF_PROGRESSION_FEATURES, dimension=self.__dimension,
                dropout=dropout, device=device, dtype=dtype)
        elif position_encoder == 'position_embedding':
            self.position_encoder = PositionEmbedding(
                max_length=MAX_LENGTH_OF_PROGRESSION_FEATURES, dimension=self.__dimension,
                dropout=dropout, device=device, dtype=dtype)
        else:
            raise NotImplementedError(position_encoder)

        self.candidates_embedding = nn.Embedding(
            NUM_TYPES_OF_ACTIONS + 1, self.__dimension, padding_idx=NUM_TYPES_OF_ACTIONS,
            device=device, dtype=dtype)

        encoder_layer = nn.TransformerEncoderLayer(
            self.__dimension, num_heads, dim_feedforward=dim_feedforward,
            activation=activation_function, dropout=dropout, batch_first=True, device=device,
            dtype=dtype)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.checkpointing = checkpointing

    def forward(
            self, sparse: Tensor, numeric: Tensor, progression: Tensor,
            candidates: Tensor) -> Tensor:
        assert sparse.dim() == 2
        batch_size = sparse.size(0)
        assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
        assert numeric.dim() == 2
        assert numeric.size(0) == batch_size
        assert numeric.size(1) == NUM_NUMERIC_FEATURES
        assert progression.dim() == 2
        assert progression.size(0) == batch_size
        assert progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
        assert candidates.dim() == 2
        assert candidates.size(0) == batch_size
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES

        sparse = self.sparse_embedding(sparse)

        device = sparse.device
        dtype = sparse.dtype
        numeric = numeric.to(device=torch.device('cpu'))
        _numeric = torch.zeros(
            (batch_size, NUM_NUMERIC_FEATURES, self.__dimension // 2), requires_grad=False,
            device=torch.device('cpu'), dtype=dtype)
        for i in range(batch_size):
            benchang: int = numeric[i, 0].item()
            _numeric[i, 0] = piecewise_linear_encoding(
                benchang, 0.0, self.__dimension // 2, self.__dimension // 2, torch.device('cpu'),
                dtype)
            deposites: int = numeric[i, 1].item()
            _numeric[i, 1] = piecewise_linear_encoding(
                deposites, 0.0, self.__dimension // 2, self.__dimension // 2, torch.device('cpu'),
                dtype)
            for seat in range(4):
                score: int = numeric[i, 2 + seat].item()
                _numeric[i, 2 + seat] = piecewise_linear_encoding(
                    score, 0.0, 100000.0, self.__dimension // 2, torch.device('cpu'), dtype)
        _numeric = _numeric.to(device=device)
        numeric_embedding = self.numeric_embedding(
            torch.arange(NUM_NUMERIC_FEATURES, device=device, dtype=torch.int32))
        numeric_embedding = numeric_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        numeric = torch.cat((_numeric, numeric_embedding), 2)

        progression = self.progression_embedding(progression)
        progression = self.position_encoder(progression)

        candidates = self.candidates_embedding(candidates)

        embedding = torch.cat((sparse, numeric, progression, candidates), 1)

        if self.checkpointing:
            encoder_layers = self.encoder.layers
            encode = checkpoint_sequential(encoder_layers, len(encoder_layers), embedding)
        else:
            encode: Tensor = self.encoder(embedding)

        return encode
