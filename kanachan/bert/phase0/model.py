#!/usr/bin/env python3

from torch import nn
from kanachan.bert.encoder import Encoder
from kanachan.bert.phase0.decoder import Decoder


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super(Model, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder

    def forward(self, x):
        encode = self.__encoder(x)
        prediction = self.__decoder(encode)
        return prediction
