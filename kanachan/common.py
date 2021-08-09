#!/usr/bin/env python3

import pathlib
import datetime
import logging
import sys
from typing import Union
from torch import nn
import torch.utils.data
from kanachan import common


NUM_SPARSE_FEATURES = 32756
MAX_NUM_ACTIVE_SPARSE_FEATURES = 123
NUM_FLOAT_FEATURES = 6
NUM_FEATURES = 32762

NUM_POST_ZIMO_ACTIONS = 224


def initialize_logging(path: pathlib.Path) -> None:
    fmt = '%(asctime)s %(filename)s:%(lineno)d:%(levelname)s: %(message)s'
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(path, encoding='UTF-8')
    handlers = (console_handler, file_handler)
    logging.basicConfig(format=fmt, level=logging.INFO, handlers=handlers)


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
    def __init__(self, dimension, num_heads, num_layers, sparse) -> None:
        super(Encoder, self).__init__()
        self.__embedding = nn.Embedding(
            NUM_SPARSE_FEATURES + 1, dimension,
            padding_idx=NUM_SPARSE_FEATURES, sparse=sparse)
        layer = nn.TransformerEncoderLayer(dimension, num_heads, batch_first=True)
        self.__encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, sparse_feature, float_feature):
        embedding = self.__embedding(sparse_feature)
        feature = torch.cat((embedding, float_feature), 1)
        return self.__encoder(feature)


class DecoderBase(nn.Module):
    def __init__(self, num_actions, dimension, num_heads, num_layers, sparse) -> None:
        super(DecoderBase, self).__init__()
        self.__embedding = nn.Embedding(num_actions, dimension, sparse=sparse)
        layer = nn.TransformerDecoderLayer(dimension, num_heads, batch_first=True)
        self.__decoder = nn.TransformerDecoder(layer, num_layers)

    def forward(self, encode, action):
        embedding = self.__embedding(action)
        decode = self.__decoder(embedding, encode)
        return decode


def get_validation_loss(
        data: pathlib.Path, iterator_adaptor, model, loss_function,
        batch_size) -> float:
    num_batches = 0
    validation_loss = 0.0
    with Dataset(data, iterator_adaptor) as dataset:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for sparse_feature, float_feature, action, y in data_loader:
                prediction = model(sparse_feature, float_feature, action)
                validation_loss += loss_function(prediction, y).item()
                num_batches += 1

    validation_loss /= num_batches

    return validation_loss


def training_epoch(
        training_data: pathlib.Path, validation_data: pathlib.Path,
        iterator_adaptor, model, loss_function, optimizer, batch_size,
        writer, epoch):
    logging.info(f'The {epoch}-th epoch starts.')

    start_time = datetime.datetime.now()

    num_batches = 0
    training_loss = 0.0
    with Dataset(training_data, iterator_adaptor) as dataset:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for batch, (sparse_feature, float_feature, action, y) in enumerate(data_loader):
            prediction = model(sparse_feature, float_feature, action)
            loss = loss_function(prediction, y)
            training_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(f'epoch = {epoch}, batch = {batch}, loss = {loss.item()}')

            num_batches = batch + 1

    elapsed_time = datetime.datetime.now() - start_time
    logging.info(f'The {epoch}-th epoch has finished (elapsed time = {elapsed_time}).')

    training_loss /= num_batches
    logging.info(f'epoch = {epoch}, training loss = {training_loss}')
    writer.add_scalar('Training epoch loss', training_loss, epoch)

    if validation_data is not None:
        validation_loss = get_validation_loss(
            validation_data, iterator_adaptor, model, loss_function, batch_size)
        logging.info(f'epoch = {epoch}, validation loss = {validation_loss}')
        writer.add_scalar('Validation epoch loss', validation_loss, epoch)
        return validation_loss

    return None
