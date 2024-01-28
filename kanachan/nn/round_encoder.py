import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint_sequential
from kanachan.constants import (
    ROUND_NUM_TYPES_OF_SPARSE_FEATURES, ROUND_NUM_SPARSE_FEATURES, ROUND_NUM_NUMERIC_FEATURES)
from kanachan import piecewise_linear_encoding


class RoundEncoder(nn.Module):
    def __init__(
            self, *, position_encoder: str, dimension: int, num_heads: int, dim_feedforward: int,
            num_layers: int, activation_function: str, dropout: float, checkpointing: bool,
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
            ROUND_NUM_TYPES_OF_SPARSE_FEATURES, self.__dimension, device=device, dtype=dtype)

        numeric_embedding = torch.randn(
            ROUND_NUM_NUMERIC_FEATURES, self.__dimension // 2, requires_grad=True, device=device,
            dtype=dtype)
        self.numeric_embedding = nn.Parameter(numeric_embedding, requires_grad=True)

        encoder_layer = nn.TransformerEncoderLayer(
            self.__dimension, num_heads, dim_feedforward=dim_feedforward,
            activation=activation_function, dropout=dropout, batch_first=True, device=device,
            dtype=dtype)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.checkpointing = checkpointing

    def forward(self, sparse: Tensor, numeric: Tensor) -> Tensor:
        assert sparse.dim() == 2
        batch_size = sparse.size(0)
        assert sparse.size(1) == ROUND_NUM_SPARSE_FEATURES
        assert numeric.dim() == 2
        assert numeric.size(0) == batch_size
        assert numeric.size(1) == ROUND_NUM_NUMERIC_FEATURES

        sparse = self.sparse_embedding(sparse)

        device = sparse.device
        dtype = sparse.dtype
        _numeric = torch.zeros(
            (batch_size, ROUND_NUM_NUMERIC_FEATURES, self.__dimension // 2), requires_grad=False,
            device=device, dtype=dtype)
        for i in range(batch_size):
            benchang: int = numeric[i, 0].item()
            _numeric[i, 0] = piecewise_linear_encoding(
                benchang, 0.0, self.__dimension // 2, self.__dimension // 2, device, dtype)
            deposites: int = numeric[i, 1].item()
            _numeric[i, 1] = piecewise_linear_encoding(
                deposites, 0.0, self.__dimension // 2, self.__dimension // 2, device, dtype)
            for seat in range(4):
                score: int = numeric[i, 2 + seat].item()
                _numeric[i, 2 + seat] = piecewise_linear_encoding(
                    score, 0.0, 100000.0, self.__dimension // 2, device, dtype)
        numeric_embedding = self.numeric_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        numeric = torch.cat((_numeric, numeric_embedding), 2)

        embedding = torch.cat((sparse, numeric), 1)

        if self.checkpointing:
            encoder_layers = self.encoder.layers
            encode = checkpoint_sequential(encoder_layers, len(encoder_layers), embedding)
        else:
            encode = self.encoder(embedding)

        return encode
