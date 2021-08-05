#!/usr/bin/env python3

import pathlib
import datetime
import logging
from typing import Union
from torch import nn
import torch.utils.data
from kanachan import common


NUM_COMMON_SPARSE_FEATURES = 32756
MAX_NUM_ACTIVE_COMMON_SPARSE_FEATURES = 123
NUM_COMMON_FLOAT_FEATURES = 6
NUM_COMMON_FEATURES = 32762


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, path: Union[str, pathlib.Path], iterator_adaptor) -> None:
        super(Dataset, self).__init__()
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not path.exists():
            raise RuntimeError(f'{path}: does not exist.')
        self.__path = path
        self.__iterator_adaptor = iterator_adaptor

    def __enter__(self):
        self.__fp = open(self.__path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__fp.close()

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError
        return self.__iterator_adaptor(self.__fp)


class Encoder(nn.Module):
    def __init__(self, dimension, num_heads, num_layers, dtype) -> None:
        super(Encoder, self).__init__()
        self.__embedding = nn.Embedding(
            NUM_COMMON_SPARSE_FEATURES + 1, dimension,
            padding_idx=NUM_COMMON_SPARSE_FEATURES, dtype=dtype)
        layer = nn.TransformerEncoderLayer(
            dimension, num_heads, batch_first=True, dtype=dtype)
        self.__encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, sparse_feature, float_feature):
        embedding = self.__embedding(sparse_feature)
        feature = torch.cat((embedding, float_feature), 1)
        return self.__encoder(feature)


class Decoder(nn.Module):
    def __init__(self, num_actions, dimension, num_heads, num_layers, dtype) -> None:
        super(Decoder, self).__init__()
        self.__embedding = nn.Embedding(num_actions, dimension, dtype=dtype)
        layer = nn.TransformerDecoderLayer(
            dimension, num_heads, batch_first=True, dtype=dtype)
        self.__decoder = nn.TransformerDecoder(layer, num_layers)
        self.__linear = nn.Linear(dimension, 1, dtype=dtype)

    def forward(self, encode, action):
        embedding = self.__embedding(action)
        decode = self.__decoder(embedding, encode)
        decode = torch.flatten(decode, start_dim=1)
        output = self.__linear(decode)
        return torch.flatten(output)


def training_epoch(path: pathlib.Path, iterator_adaptor, encoder, decoder, optimizer, batch_size, epoch=None):
    if epoch is None:
        logging.info('A new epoch starts.')
    else:
        logging.info(f'The {epoch}-th epoch starts.')

    start_time = datetime.datetime.now()

    with Dataset(path, iterator_adaptor) as dataset:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for batch, (sparse_feature, float_feature, action, y) in enumerate(data_loader):
            encode = encoder(sparse_feature, float_feature)
            prediction = decoder(encode, action)
            loss_function = nn.MSELoss()
            loss = loss_function(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch is None:
                logging.info(f'batch = {batch}, loss = {loss.item()}')
            else:
                logging.info(f'epoch = {epoch}, batch = {batch}, loss = {loss.item()}')

    elapsed_time = datetime.datetime.now() - start_time
    if epoch is None:
        logging.info(f'An epoch has finished (elapsed time = {elapsed_time}).')
    else:
        logging.info(f'The {epoch}-th epoch has finished (elapsed time = {elapsed_time}).')


def validate(path: pathlib.Path, iterator_adaptor, model, batch_size):
    validation_loss = 0.0
    num_samples = 0

    with Dataset(path, iterator_adaptor) as dataset:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for x, y, z in data_loader:
                output = model(x, y)
                validation_loss += nn.MSELoss(output, z).item()
                num_samples += 1

    validation_loss /= num_samples
    logging.info(f'validation loss = {validation_loss}')
