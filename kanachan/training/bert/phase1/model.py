import torch
from torch import nn
from kanachan.nn import Encoder
from kanachan.training.bert.phase1.decoder import Decoder


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self, sparse: torch.Tensor, numeric: torch.Tensor, progression: torch.Tensor,
            candidates: torch.Tensor) -> torch.Tensor:
        encode: torch.Tensor = self.encoder(sparse, numeric, progression, candidates)
        weights: torch.Tensor = self.decoder(candidates, encode)
        return weights
