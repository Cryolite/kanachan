#!/usr/bin/env python3

import torch
from torch import nn
from kanachan.bert.encoder import Encoder
from kanachan.bert.phase1.decoder import Decoder


class Model(nn.Module):
    def __init__(
            self, encoder: Encoder, decoder: Decoder,
            *, freeze_encoder: bool) -> None:
        super(Model, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder
        self.__freeze_encoder = freeze_encoder

    def mode(self, mode: bool) -> None:
        if mode not in ('training', 'validation', 'prediction'):
            raise ValueError(mode)

    def forward(self, x):
        if self.__freeze_encoder:
            with torch.no_grad():
                encode = self.__encoder(x)
        else:
            encode = self.__encoder(x)
        prediction = self.__decoder(encode)
        return prediction
