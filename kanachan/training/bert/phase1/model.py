#!/usr/bin/env python3

import torch
from torch import nn
from kanachan.training.bert.encoder import Encoder
from kanachan.training.bert.phase1.decoder import Decoder


class Model(nn.Module):
    def __init__(
            self, encoder: Encoder, decoder: Decoder,
            *, freeze_encoder: bool=False) -> None:
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.freeze_encoder = freeze_encoder

    def mode(self, mode: str) -> None:
        if mode not in ('training', 'validation', 'prediction'):
            raise ValueError(mode)

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                encode = self.encoder(x)
        else:
            encode = self.encoder(x)
        prediction = self.decoder(encode)
        return prediction
