#!/usr/bin/env python3

from torch import nn
from kanachan.training.aiayn.encoder import Encoder
from kanachan.training.aiayn.decoder import Decoder


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super(Model, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder

    def forward(self, x):
        encode = self.__encoder(x[:-1])
        prediction = self.__decoder(encode, x[-1])
        return prediction
