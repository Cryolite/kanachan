#!/usr/bin/env python3

import pathlib
import datetime
import logging
import math
import sys
from typing import Union
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import IterableDataset
from kanachan import common


def initialize_logging(path: pathlib.Path) -> None:
    fmt = '%(asctime)s %(filename)s:%(lineno)d:%(levelname)s: %(message)s'
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(path, encoding='UTF-8')
    handlers = (console_handler, file_handler)
    logging.basicConfig(format=fmt, level=logging.INFO, handlers=handlers)


class Dataset(IterableDataset):
    def __init__(self, path: Union[str, pathlib.Path], iterator_adaptor) -> None:
        super(Dataset, self).__init__()
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not path.exists():
            raise RuntimeError(f'{path}: does not exist.')
        self.__path = path
        self.__iterator_adaptor = iterator_adaptor

    def __iter__(self):
        return self.__iterator_adaptor(self.__path)


class PositionalEncoding(nn.Module):
    def __init__(self, max_length: int, dimension: int, dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        pe = torch.zeros(max_length, dimension)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, num_dimensions, num_heads, num_layers, sparse) -> None:
        super(Encoder, self).__init__()
        self.__num_dimensions = num_dimensions
        self.__embedding = nn.Embedding(
            NUM_SPARSE_FEATURES + 1, self.__num_dimensions,
            padding_idx=NUM_SPARSE_FEATURES, sparse=sparse)
        self.__positional_embedding = nn.Embedding(
            NUM_TYPES_OF_POSITIONAL_FEATURES + 1, self.__num_dimensions,
            padding_idx=NUM_TYPES_OF_POSITIONAL_FEATURES, sparse=sparse)
        self.__positional_encoding = PositionalEncoding(
            MAX_LENGTH_OF_POSITIONAL_FEATURES, self.__num_dimensions)
        layer = nn.TransformerEncoderLayer(
            self.__num_dimensions, num_heads, batch_first=True)
        self.__encoder = nn.TransformerEncoder(layer, num_layers)

    @property
    def num_dimensions(self) -> int:
        return self.__num_dimensions

    def forward(self, x):
        sparse, numeric, positional = x
        embedding = self.__embedding(sparse)
        positional_embedding = self.__positional_embedding(positional)
        positional_embedding = self.__positional_encoding(positional_embedding)
        feature = torch.cat((embedding, numeric, positional_embedding), 1)
        return self.__encoder(feature)


class DecoderBase(nn.Module):
    def __init__(self, dimension, num_heads, num_layers, sparse) -> None:
        super(DecoderBase, self).__init__()
        self.__embedding = nn.Embedding(NUM_ACTIONS, dimension, sparse=sparse)
        layer = nn.TransformerDecoderLayer(dimension, num_heads, batch_first=True)
        self.__decoder = nn.TransformerDecoder(layer, num_layers)

    def forward(self, encode, action):
        embedding = self.__embedding(action)
        decode = self.__decoder(embedding, encode)
        decode = torch.flatten(decode, start_dim=1)
        return decode


def get_validation_loss(
        data: pathlib.Path, iterator_adaptor, num_workers, model,
        loss_function, batch_size, device) -> float:
    num_batches = 0
    validation_loss = 0.0
    dataset = Dataset(data, iterator_adaptor)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True)
    with torch.no_grad():
        for x, y in data_loader:
            if device != 'cpu':
                x = tuple(e.to(device=device) for e in x)
                y = y.to(device=device)
            prediction = model(x)
            validation_loss += loss_function(prediction, y).item()
            num_batches += 1

    validation_loss /= num_batches

    return validation_loss


def training_epoch(
        training_data: pathlib.Path, validation_data: pathlib.Path,
        iterator_adaptor, num_workers, model, loss_function, optimizer,
        grad_scaler, batch_size, device, dtype, writer, epoch):
    logging.info(f'The {epoch}-th epoch starts.')

    start_time = datetime.datetime.now()

    num_batches = 0
    training_loss = 0.0
    dataset = Dataset(training_data, iterator_adaptor)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True)
    for batch, (x, y) in enumerate(data_loader):
        if device != 'cpu':
            x = tuple(e.to(device=device) for e in x)
            y = y.to(device=device)

        with amp.autocast(enabled=(dtype == torch.float32)):
            prediction = model(x)
            loss = loss_function(prediction, y)
            if math.isnan(loss.item()):
                raise RuntimeError('Loss becomes NaN in training.')

        training_loss += loss.item()

        optimizer.zero_grad()
        if dtype == torch.float32:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
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
            validation_data, iterator_adaptor, num_workers, model, loss_function,
            batch_size, device)
        logging.info(f'epoch = {epoch}, validation loss = {validation_loss}')
        writer.add_scalar('Validation epoch loss', validation_loss, epoch)
        return validation_loss

    return None
