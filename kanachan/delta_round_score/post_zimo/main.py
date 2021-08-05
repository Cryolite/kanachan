#!/usr/bin/env python3

import pathlib
import logging
import itertools
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from kanachan import common
from kanachan.delta_round_score.post_zimo.iterator_adaptor import IteratorAdaptor


if __name__ == '__main__':
    fmt = '%(asctime)s %(filename)s:%(lineno)d:%(levelname)s: %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)

    if (len(sys.argv) != 2):
        raise RuntimeError(len(sys.argv))

    path = pathlib.Path(sys.argv[1])
    if not path.exists():
        raise RuntimeError(f'{path}: does not exist.')

    dtype = torch.float32
    num_actions = 224
    dimension = 256
    num_heads = 8
    num_layers = 5
    batch_size = 32

    encoder = common.Encoder(dimension, num_heads, num_layers, dtype)
    decoder = common.Decoder(224, dimension, num_heads, num_layers, dtype)

    parameters = itertools.chain(encoder.parameters(), decoder.parameters())
    optimizer = torch.optim.Adam(parameters)

    iterator_adaptor = lambda fp: IteratorAdaptor(fp, dimension, dtype)
    common.training_epoch(path, iterator_adaptor, encoder, decoder, optimizer, batch_size, epoch=0)
