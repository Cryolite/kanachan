#!/usr/bin/env python3

import torch
from kanachan.constants import MAX_NUM_ACTION_CANDIDATES
from kanachan.bert.phase2.decoder_base import DecoderBase


class PredictionDecoder(DecoderBase):
    def __init__(self, *, **kwargs) -> None:
        super(PredictionDecoder, self).__init__(**kwargs)

    def forward(self, encode):
        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
        decode = self._semifinal_linear(encode)
        decode = self._semifinal_dropout(decode)
        decode = self._semifinal_activation(decode)

        prediction = self._final_linear(decode)
        prediction = torch.squeeze(prediction, dim=2)
        return prediction
