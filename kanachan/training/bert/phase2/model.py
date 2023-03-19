#!/usr/bin/env python3

import torch
from torch import nn
from kanachan.training.bert.encoder import Encoder
from kanachan.training.bert.phase2.decoder import Decoder


class Model(nn.Module):
    def __init__(
            self, encoder: Encoder, decoder: Decoder,
            *, freeze_encoder: bool) -> None:
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.freeze_encoder = freeze_encoder
        self._mode = 'training'

    def mode(self, mode: str) -> None:
        if mode not in ('training', 'validation', 'prediction'):
            raise ValueError(mode)
        self.decoder.mode(mode)
        self._mode = mode

    def forward(self, x):
        if self._mode == 'prediction':
            encode = self.encoder(x)
        else:
            if self.freeze_encoder:
                with torch.no_grad():
                    encode = self.encoder(x[:-1])
            else:
                encode = self.encoder(x[:-1])
            encode = (encode, x[-1])
        prediction = self.decoder(encode)
        return prediction
