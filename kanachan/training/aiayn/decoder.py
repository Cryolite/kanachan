#!/usr/bin/env python3

import torch
from torch import nn
from kanachan.training.constants import (NUM_TYPES_OF_ACTIONS,)


class Decoder(nn.Module):
    def __init__(
            self, num_dimensions: int, num_heads: int, num_layers: int,
            num_final_dimensions: int=2048, dropout: float=0.1,
            sparse: bool=False) -> None:
        super(Decoder, self).__init__()

        self.__candidates_embedding = nn.Embedding(
            NUM_TYPES_OF_ACTIONS + 2, num_dimensions,
            padding_idx=NUM_TYPES_OF_ACTIONS + 1, sparse=sparse)
        self.__candidates_dropout = nn.Dropout(p=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            num_dimensions, num_heads, dropout=dropout, batch_first=True)
        self.__decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # The final layer is position-wise feed-forward network.
        self.__semifinal_linear = nn.Linear(num_dimensions, num_final_dimensions)
        self.__final_activation = nn.ReLU()
        self.__final_dropout = nn.Dropout(p=dropout)
        self.__final_linear = nn.Linear(num_final_dimensions, 1)

    def forward(self, encode, candidates):
        candidates_embedding = self.__candidates_embedding(candidates)
        candidates_embedding = self.__candidates_dropout(candidates_embedding)

        decode = self.__decoder(candidates_embedding, encode)

        output = self.__semifinal_linear(decode)
        output = self.__final_activation(output)
        output = self.__final_dropout(output)

        prediction = self.__final_linear(output)
        prediction = torch.squeeze(prediction, dim=2)
        return prediction
