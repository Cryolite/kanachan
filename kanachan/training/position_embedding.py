import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(
            self, *, max_length: int, dimension: int, dropout: float,
            device: torch.device, dtype: torch.dtype):
        super(PositionEmbedding, self).__init__()

        self.position_embedding = nn.Embedding(max_length, dimension, device=device, dtype=dtype)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        x += self.position_embedding(torch.arange(x.size(1), device=x.device, dtype=x.dtype))
        return self.dropout(x)
