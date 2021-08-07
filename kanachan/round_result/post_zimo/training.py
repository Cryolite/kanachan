#!/usr/bin/env python3

import re
import datetime
import pathlib
import logging
import itertools
import sys
from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from kanachan import common
import kanachan.round_result.common as round_result_common
from kanachan.delta_round_score.post_zimo.iterator_adaptor import IteratorAdaptor


class Decoder(common.DecoderBase):
    def __init__(self, dimension, num_heads, num_layers) -> None:
        super(Decoder, self).__init__(
            common.NUM_POST_ZIMO_ACTIONS, dimension, num_heads, num_layers)
        self.__activation = nn.ReLU()
        self.__full_connection = nn.Linear(dimension, round_result_common.NUM_ROUND_RESULT_CATEGORIES)
        self.__softmax = nn.Softmax(dim=1)

    def forward(self, encode, action):
        decode = super(Decoder, self).forward(encode, action)
        decode = self.__activation(decode)
        decode = torch.flatten(decode, start_dim=1)
        prediction = self.__full_connection(decode)
        return prediction


class Model(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(Model, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder

    def forward(self, sparse_feature, float_feature, action):
        encode = self.__encoder(sparse_feature, float_feature)
        prediction = self.__decoder(encode, action)
        return prediction


if __name__ == '__main__':
    fmt = '%(asctime)s %(filename)s:%(lineno)d:%(levelname)s: %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)

    ap = ArgumentParser(
        description='Train a model predicting the round result by post-zimo action.')
    ap.add_argument_group(title='Data')
    ap.add_argument('--training-data', type=pathlib.Path, required=True, metavar='PATH')
    ap.add_argument('--validation-data', type=pathlib.Path, metavar='PATH')
    ap.add_argument_group(title='Device')
    ap.add_argument('--device', default='cpu', metavar='DEV')
    ap.add_argument(
        '--dtype', default='float32', choices=('float16','float32', 'float64'),
        metavar='TYPE')
    ap.add_argument_group(title='Model')
    ap.add_argument('--dimension', default=128, type=int, metavar='N')
    ap.add_argument('--num-heads', default=8, type=int, metavar='N')
    ap.add_argument('--num-layers', default=5, type=int, metavar='N')
    ap.add_argument('--sparse', action='store_true')
    ap.add_argument_group(title='Training')
    ap.add_argument('--batch-size', default=32, type=int, metavar='N')
    ap.add_argument('--max-epoch', default=100, type=int, metavar='N')
    ap.add_argument('--optimizer', default='adam', choices=('adam', 'sgd'))
    ap.add_argument('--learning-rate', type=float, metavar='LR')
    ap.add_argument_group(title='Output')
    ap.add_argument('--output-prefix', type=pathlib.Path, required=True, metavar='PATH')

    config = ap.parse_args()
    if re.search('^(?:cpu)|(?:cuda(?::\\d))', config.device) is None:
        raise RuntimeError(f'{config.device}: invalid device.')
    if not config.training_data.exists():
        raise RuntimeError(f'{config.training_data}: does not exist.')
    if config.validation_data is not None and not config.validation_data.exists():
        raise RuntimeError(f'{config.validation_data}: does not exist.')
    if config.dtype == 'float16':
        dtype = torch.float16
    elif config.dtype == 'float32':
        dtype = torch.float32
    else:
        assert(config.dtype == 'float64')
        dtype = torch.float64
    if config.dimension < 1:
        raise RuntimeError(f'{config.dimension}: invalid dimension.')

    iterator_adaptor = lambda fp: round_result_common.IteratorAdaptor(
        fp, config.dimension, device=config.device, dtype=dtype)

    encoder = common.Encoder(config.dimension, config.num_heads, config.num_layers, sparse=config.sparse)
    decoder = Decoder(config.dimension, config.num_heads, config.num_layers)
    model = Model(encoder, decoder)
    model.to(device=config.device, dtype=dtype)

    loss_function = nn.CrossEntropyLoss()

    if config.optimizer is None:
        if config.sparse:
            if config.learning_rate is None:
                raise RuntimeError(
                    'Use `--optimizer sgd` with `--learning-rate`, or do not use `--sparse`')
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        else:
            if config.learning_rate is None:
                optimizer = torch.optim.Adam(model.parameters())
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'adam':
        if config.sparse:
            raise RuntimeError(
                'Do not use `--sparse`, or use `--optimizer sgd` with `--learning-rate`.')
        else:
            if config.learning_rate is None:
                optimizer = torch.optim.Adam(model.parameters())
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        assert(config.optimizer == 'sgd')
        if config.learning_rate is None:
            raise RuntimeError('Set `--learning-rate` for SGD.')
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    now = datetime.datetime.now()
    tensorboard_path = config.output_prefix / 'tensorboard' / now.strftime('%Y-%m-%d-%H-%M-%S')

    logging.info(f'training data: {config.training_data}')
    if config.validation_data is None:
        logging.info(f'Validation data: N/A')
    else:
        logging.info(f'Validation data: {config.validation_data}')
    logging.info(f'Device: {config.device}')
    logging.info(f'dtype: {dtype}')
    logging.info(f'Dimension: {config.dimension}')
    logging.info(f'# of heads: {config.num_heads}')
    logging.info(f'# of layers: {config.num_layers}')
    logging.info(f'Sparse: {config.sparse}')
    logging.info(f'Batch size: {config.batch_size}')
    logging.info(f'Max epoch: {config.max_epoch}')
    logging.info(f'Optimizer: {config.optimizer}')
    logging.info(f'Learning rate: {config.learning_rate}')
    logging.info(f'Output prefix: {config.output_prefix}')

    with tensorboard.SummaryWriter(log_dir=tensorboard_path) as writer:
        for i in range(config.max_epoch):
            common.training_epoch(
                config.training_data, config.validation_data, iterator_adaptor,
                model, loss_function, optimizer, config.batch_size, writer, i)
