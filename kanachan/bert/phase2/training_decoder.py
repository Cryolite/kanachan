#!/usr/bin/env python3

import torch
from kanachan.constants import MAX_NUM_ACTION_CANDIDATES
from kanachan.bert.phase2.decoder_base import DecoderBase


class TrainingDecoder(DecoderBase):
    def __init__(self, *, **kwargs) -> None:
        super(TrainingDecoder, self).__init__(**kwargs)

    def forward(self, encode):
        encode, index = encode
        assert(torch.all(0 <= index))
        assert(torch.all(index < MAX_NUM_ACTION_CANDIDATES))
        rows = torch.arange(encode.size()[0])
        encode = encode[rows, -MAX_NUM_ACTION_CANDIDATES + index]
        decode = self._semifinal_linear(encode)
        decode = self._semifinal_dropout(decode)
        decode = self._semifinal_activation(decode)

        prediction = self._final_linear(decode)
        prediction = torch.squeeze(prediction, dim=1)
        return prediction
