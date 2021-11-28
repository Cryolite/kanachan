#!/usr/bin/env python3

import torch
from torch import nn
from kanachan.bert.encoder import Encoder
from kanachan.bert.phase2.decoder import Decoder


class Model(nn.Module):
    def __init__(
            self, encoder: Encoder, decoder: Decoder,
            *, freeze_encoder: bool) -> None:
        super(Model, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self.__freeze_encoder = freeze_encoder
        self.__mode = 'training'

    def mode(self, mode: str) -> None:
        if mode not in ('training', 'validation', 'prediction'):
            raise ValueError(mode)
        self._decoder.mode(mode)
        self.__mode = mode

    def forward(self, x):
        if self.__mode == 'prediction':
            encode = self._encoder(x)
        else:
            if self.__freeze_encoder:
                with torch.no_grad():
                    encode = self._encoder(x[:-1])
            else:
                encode = self._encoder(x[:-1])
            encode = (encode, x[-1])
        prediction = self._decoder(encode)
        return prediction
