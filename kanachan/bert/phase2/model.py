#!/usr/bin/env python3

from torch import nn
from kanachan.bert.encoder import Encoder
from kanachan.bert.phase2.training_decoder import TrainingDecoder


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: TrainingDecoder) -> None:
        super(Model, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder

    def forward(self, x):
        encode = self.__encoder(x[:-1])
        encode = (encode, x[-1])
        prediction = self.__decoder(encode)
        return prediction
